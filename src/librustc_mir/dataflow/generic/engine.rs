//! A solver for dataflow problems.

use std::ffi::OsString;
use std::path::PathBuf;
use std::fs;

use rustc::hir::def_id::DefId;
use rustc::mir::{self, traversal, BasicBlock, Location};
use rustc::ty::TyCtxt;
use rustc_data_structures::work_queue::WorkQueue;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use syntax::symbol::{sym, Symbol};
use syntax::ast;

use super::graphviz;
use super::{Analysis, GenKillAnalysis, GenKillSet, Results};

/// A solver for dataflow problems.
pub struct Engine<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    def_id: DefId,
    dead_unwinds: Option<&'a BitSet<BasicBlock>>,
    analysis: A,

    /// Cached, cumulative transfer functions for each block.
    ///
    /// These are only computable for gen-kill problems.
    trans_for_block: Option<IndexVec<BasicBlock, GenKillSet<A::Idx>>>,
}

impl<A> Engine<'a, 'tcx, A>
where
    A: GenKillAnalysis<'tcx>,
{
    /// Creates a new `Engine` to solve a gen-kill dataflow problem.
    pub fn new_gen_kill(
        tcx: TyCtxt<'tcx>,
        body: &'a mir::Body<'tcx>,
        def_id: DefId,
        analysis: A,
    ) -> Self {
        let bits_per_block = analysis.bits_per_block(body);
        let mut trans_for_block =
            IndexVec::from_elem(GenKillSet::identity(bits_per_block), body.basic_blocks());

        // Compute cumulative block transfer functions.
        //
        // FIXME: we may want to skip this if the MIR is acyclic, since we will never access a
        // block transfer function more than once.

        for (block, block_data) in body.basic_blocks().iter_enumerated() {
            let trans = &mut trans_for_block[block];

            for (i, statement) in block_data.statements.iter().enumerate() {
                let loc = Location { block, statement_index: i };
                analysis.before_statement_effect(trans, statement, loc);
                analysis.statement_effect(trans, statement, loc);
            }

            if let Some(terminator) = &block_data.terminator {
                let loc = Location { block, statement_index: block_data.statements.len() };
                analysis.before_terminator_effect(trans, terminator, loc);
                analysis.terminator_effect(trans, terminator, loc);
            }
        }

        Self::new(tcx, body, def_id, analysis, Some(trans_for_block))
    }
}

impl<A> Engine<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// Creates a new `Engine` to solve a dataflow problem with an arbitrary transfer
    /// function.
    ///
    /// Gen-kill problems should use `new_gen_kill`, which will coalesce transfer functions for
    /// better performance.
    pub fn new_generic(
        tcx: TyCtxt<'tcx>,
        body: &'a mir::Body<'tcx>,
        def_id: DefId,
        analysis: A,
    ) -> Self {
        Self::new(tcx, body, def_id, analysis, None)
    }

    fn new(
        tcx: TyCtxt<'tcx>,
        body: &'a mir::Body<'tcx>,
        def_id: DefId,
        analysis: A,
        trans_for_block: Option<IndexVec<BasicBlock, GenKillSet<A::Idx>>>,
    ) -> Self {
        Engine {
            analysis,
            tcx,
            body,
            def_id,
            dead_unwinds: None,
            trans_for_block,
        }
    }

    pub fn dead_unwinds(mut self, dead_unwinds: &'a BitSet<BasicBlock>) -> Self {
        self.dead_unwinds = Some(dead_unwinds);
        self
    }

    pub fn iterate_to_fixpoint(self) -> Results<'tcx, A> {
        // Initialize the entry sets for each block.

        let bits_per_block = self.analysis.bits_per_block(self.body);
        let bottom_value_set = if A::BOTTOM_VALUE == true {
            BitSet::new_filled(bits_per_block)
        } else {
            BitSet::new_empty(bits_per_block)
        };

        let mut entry_sets = IndexVec::from_elem(bottom_value_set, self.body.basic_blocks());
        self.analysis.initialize_start_block(self.body, &mut entry_sets[mir::START_BLOCK]);

        // To improve performance, we check for the existence of cached block transfer functions
        // *outside* the loop in `_iterate_to_fixpoint` below.
        if let Some(trans_for_block) = &self.trans_for_block {
            self._iterate_to_fixpoint(
                bits_per_block,
                &mut entry_sets,
                |state, bb| trans_for_block[bb].apply(state),
            );
        } else {
            self._iterate_to_fixpoint(
                bits_per_block,
                &mut entry_sets,
                |state, bb| {
                    let block_data = &self.body[bb];
                    apply_whole_block_effect(&self.analysis, state, bb, block_data);
                }
            );
        }

        let Engine { tcx, def_id, body, analysis, trans_for_block, .. } = self;
        let results = Results { analysis, entry_sets };

        let res = write_graphviz_results(tcx, def_id, body, &results, trans_for_block);
        if let Err(e) = res {
            warn!("Failed to write graphviz dataflow results: {}", e);
        }

        results
    }

    /// Helper function that propagates dataflow state into graph succesors until fixpoint is
    /// reached.
    fn _iterate_to_fixpoint(
        &self,
        bits_per_block: usize,
        entry_sets: &mut IndexVec<BasicBlock, BitSet<A::Idx>>,
        apply_block_effect: impl Fn(&mut BitSet<A::Idx>, BasicBlock),
    ) {
        let body = self.body;
        let mut state = BitSet::new_empty(bits_per_block);

        let mut dirty_queue: WorkQueue<BasicBlock> =
            WorkQueue::with_none(body.basic_blocks().len());

        for (bb, _) in traversal::reverse_postorder(body) {
            dirty_queue.insert(bb);
        }

        // Add blocks that are not reachable from START_BLOCK to the work queue. These blocks will
        // be processed after the ones added above.
        for bb in body.basic_blocks().indices() {
            dirty_queue.insert(bb);
        }

        while let Some(bb) = dirty_queue.pop() {
            state.overwrite(&entry_sets[bb]);
            apply_block_effect(&mut state, bb);

            self.propagate_bits_into_graph_successors_of(
                entry_sets,
                &mut state,
                (bb, &body[bb]),
                &mut dirty_queue,
            );
        }
    }

    fn propagate_state_to(
        &self,
        bb: BasicBlock,
        state: &BitSet<A::Idx>,
        entry_sets: &mut IndexVec<BasicBlock, BitSet<A::Idx>>,
        dirty_queue: &mut WorkQueue<BasicBlock>,
    ) {
        let entry_set = &mut entry_sets[bb];
        let set_changed = self.analysis.join(entry_set, state);
        if set_changed {
            dirty_queue.insert(bb);
        }
    }

    fn propagate_bits_into_graph_successors_of(
        &self,
        entry_sets: &mut IndexVec<BasicBlock, BitSet<A::Idx>>,
        exit_state: &mut BitSet<A::Idx>,
        (bb, bb_data): (BasicBlock, &mir::BasicBlockData<'tcx>),
        dirty: &mut WorkQueue<BasicBlock>,
    ) {
        use mir::TerminatorKind;

        match bb_data.terminator().kind {
            | TerminatorKind::Return
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Unreachable
            => {}

            | TerminatorKind::Goto { target }
            | TerminatorKind::Assert { target, cleanup: None, .. }
            | TerminatorKind::Yield { resume: target, drop: None, .. }
            | TerminatorKind::Drop { target, location: _, unwind: None }
            | TerminatorKind::DropAndReplace { target, value: _, location: _, unwind: None }
            => self.propagate_state_to(target, exit_state, entry_sets, dirty),

            TerminatorKind::Yield { resume: target, drop: Some(drop), .. } => {
                self.propagate_state_to(target, exit_state, entry_sets, dirty);
                self.propagate_state_to(drop, exit_state, entry_sets, dirty);
            }

            | TerminatorKind::Assert { target, cleanup: Some(unwind), .. }
            | TerminatorKind::Drop { target, location: _, unwind: Some(unwind) }
            | TerminatorKind::DropAndReplace { target, value: _, location: _, unwind: Some(unwind) }
            => {
                self.propagate_state_to(target, exit_state, entry_sets, dirty);
                if self.dead_unwinds.map_or(true, |bbs| !bbs.contains(bb)) {
                    self.propagate_state_to(unwind, exit_state, entry_sets, dirty);
                }
            }

            TerminatorKind::SwitchInt { ref targets, .. } => {
                for target in targets {
                    self.propagate_state_to(*target, exit_state, entry_sets, dirty);
                }
            }

            TerminatorKind::Call { cleanup, ref destination, ref func, ref args, .. } => {
                if let Some(unwind) = cleanup {
                    if self.dead_unwinds.map_or(true, |bbs| !bbs.contains(bb)) {
                        self.propagate_state_to(unwind, exit_state, entry_sets, dirty);
                    }
                }

                if let Some((ref dest_place, dest_bb)) = *destination {
                    // N.B.: This must be done *last*, otherwise the unwind path will see the call
                    // return effect.
                    self.analysis.apply_call_return_effect(exit_state, bb, func, args, dest_place);
                    self.propagate_state_to(dest_bb, exit_state, entry_sets, dirty);
                }
            }

            TerminatorKind::FalseEdges { real_target, imaginary_target } => {
                self.propagate_state_to(real_target, exit_state, entry_sets, dirty);
                self.propagate_state_to(imaginary_target, exit_state, entry_sets, dirty);
            }

            TerminatorKind::FalseUnwind { real_target, unwind } => {
                self.propagate_state_to(real_target, exit_state, entry_sets, dirty);
                if let Some(unwind) = unwind {
                    if self.dead_unwinds.map_or(true, |bbs| !bbs.contains(bb)) {
                        self.propagate_state_to(unwind, exit_state, entry_sets, dirty);
                    }
                }
            }
        }
    }
}

/// Applies the cumulative effect of an entire block, excluding the call return effect if one
/// exists.
fn apply_whole_block_effect<A>(
    analysis: &'a A,
    state: &mut BitSet<A::Idx>,
    block: BasicBlock,
    block_data: &'a mir::BasicBlockData<'tcx>,
)
where
    A: Analysis<'tcx>,
{
    for (statement_index, statement) in block_data.statements.iter().enumerate() {
        let location = Location { block, statement_index };
        analysis.apply_before_statement_effect(state, statement, location);
        analysis.apply_statement_effect(state, statement, location);
    }

    let terminator = block_data.terminator();
    let location = Location { block, statement_index: block_data.statements.len() };
    analysis.apply_before_terminator_effect(state, terminator, location);
    analysis.apply_terminator_effect(state, terminator, location);
}

// Graphviz

/// Writes a DOT file containing the results of a dataflow analysis if the user requested it via
/// `rustc_mir` attributes.
fn write_graphviz_results<A>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    body: &mir::Body<'tcx>,
    results: &Results<'tcx, A>,
    block_transfer_functions: Option<IndexVec<BasicBlock, GenKillSet<A::Idx>>>,
) -> std::io::Result<()>
where
    A: Analysis<'tcx>,
{
    let attrs = match RustcMirAttrs::parse(tcx, def_id) {
        Ok(attrs) => attrs,

        // Invalid `rustc_mir` attrs will be reported using `span_err`.
        Err(()) => return Ok(()),
    };

    let path = match attrs.output_path(A::NAME) {
        Some(path) => path,
        None => return Ok(()),
    };

    let bits_per_block = results.analysis.bits_per_block(body);

    let mut formatter: Box<dyn graphviz::StateFormatter<'tcx, _>> = match attrs.formatter {
        Some(sym::two_phase) => Box::new(graphviz::TwoPhaseDiff::new(bits_per_block)),
        Some(sym::gen_kill) => {
            if let Some(trans_for_block) = block_transfer_functions {
                Box::new(graphviz::BlockTransferFunc::new(body, trans_for_block))
            } else {
                Box::new(graphviz::SimpleDiff::new(bits_per_block))
            }
        }

        // Default to the `SimpleDiff` output style.
        _ => Box::new(graphviz::SimpleDiff::new(bits_per_block)),
    };

    debug!("printing dataflow results for {:?} to {}", def_id, path.display());
    let mut buf = Vec::new();

    let graphviz = graphviz::Formatter::new(body, def_id, results, &mut *formatter);
    dot::render(&graphviz, &mut buf)?;
    fs::write(&path, buf)?;
    Ok(())
}

#[derive(Default)]
struct RustcMirAttrs {
    basename_and_suffix: Option<PathBuf>,
    formatter: Option<Symbol>,
}

impl RustcMirAttrs {
    fn parse(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
    ) -> Result<Self, ()> {
        let attrs = tcx.get_attrs(def_id);

        let mut result = Ok(());
        let mut ret = RustcMirAttrs::default();

        let rustc_mir_attrs = attrs
            .into_iter()
            .filter(|attr| attr.check_name(sym::rustc_mir))
            .flat_map(|attr| attr.meta_item_list().into_iter().flat_map(|v| v.into_iter()));

        for attr in rustc_mir_attrs {
            let attr_result = if attr.check_name(sym::borrowck_graphviz_postflow) {
                Self::set_field(&mut ret.basename_and_suffix, tcx, &attr, |s| {
                    let path = PathBuf::from(s.to_string());
                    match path.file_name() {
                        Some(_) => Ok(path),
                        None => {
                            tcx.sess.span_err(attr.span(), "path must end in a filename");
                            Err(())
                        }
                    }
                })
            } else if attr.check_name(sym::borrowck_graphviz_format) {
                Self::set_field(&mut ret.formatter, tcx, &attr, |s| {
                    match s {
                        sym::gen_kill | sym::two_phase => Ok(s),
                        _ => {
                            tcx.sess.span_err(attr.span(), "unknown formatter");
                            Err(())
                        }
                    }
                })
            } else {
                Ok(())
            };

            result = result.and(attr_result);
        }

        result.map(|()| ret)
    }

    fn set_field<T>(
        field: &mut Option<T>,
        tcx: TyCtxt<'tcx>,
        attr: &ast::NestedMetaItem,
        mapper: impl FnOnce(Symbol) -> Result<T, ()>,
    ) -> Result<(), ()> {
        if field.is_some() {
            tcx.sess.span_err(
                attr.span(),
                &format!("duplicate values for `{}`", attr.name_or_empty()),
            );

            return Err(());
        }

        if let Some(s) = attr.value_str() {
            *field = Some(mapper(s)?);
            Ok(())
        } else {
            tcx.sess.span_err(
                attr.span(),
                &format!("`{}` requires an argument", attr.name_or_empty()),
            );
            Err(())
        }
    }

    /// Returns the path where dataflow results should be written, or `None`
    /// `borrowck_graphviz_postflow` was not specified.
    ///
    /// This performs the following transformation to the argument of `borrowck_graphviz_postflow`:
    ///
    /// "path/suffix.dot" -> "path/analysis_name_suffix.dot"
    fn output_path(&self, analysis_name: &str) -> Option<PathBuf> {
        let mut ret = self.basename_and_suffix.as_ref().cloned()?;
        let suffix = ret.file_name().unwrap(); // Checked when parsing attrs

        let mut file_name: OsString = analysis_name.into();
        file_name.push("_");
        file_name.push(suffix);
        ret.set_file_name(file_name);

        Some(ret)
    }
}
