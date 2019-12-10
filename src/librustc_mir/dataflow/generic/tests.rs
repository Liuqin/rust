use rustc::mir::{self, Location, BasicBlock};
use rustc::ty;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;

use crate::dataflow::BottomValue;
use super::*;

#[test]
fn cursor_seek() {
    let body = dummy_body();
    let body = &body;
    let analysis = Dummy { body };

    let mut cursor = Results {
        entry_sets: analysis.entry_sets(),
        analysis,
    }.into_cursor(body);

    // Sanity check: the dummy call return effect is unique and actually being applied.

    let call_terminator_loc = Location { block: BasicBlock::from_usize(2), statement_index: 2 };
    assert!(is_call_terminator_non_diverging(body, call_terminator_loc));

    let call_return_effect =
        SeekTarget::AfterAssumeCallReturns(call_terminator_loc).effect(body).unwrap();
    assert_ne!(call_return_effect, SeekTarget::After(call_terminator_loc).effect(body).unwrap());

    cursor.seek_after(call_terminator_loc);
    assert!(!cursor.get().contains(call_return_effect));
    cursor.seek_after_assume_call_returns(call_terminator_loc);
    assert!(cursor.get().contains(call_return_effect));

    let every_target = || body
        .basic_blocks()
        .iter_enumerated()
        .flat_map(|(bb, _)| SeekTarget::iter_in_block(body, bb));

    let mut seek_to_target = |targ| {
        use SeekTarget::*;

        match targ {
            BlockStart(block) => cursor.seek_to_block_start(block),
            Before(loc) => cursor.seek_before(loc),
            After(loc) => cursor.seek_after(loc),
            AfterAssumeCallReturns(loc) => cursor.seek_after_assume_call_returns(loc),
        }

        assert_eq!(cursor.get(), &cursor.analysis().expected_state_at(targ));
    };

    // Seek *to* every possible `SeekTarget` *from* every possible `SeekTarget`.
    //
    // By resetting the cursor to `from` each time it changes, we end up checking some edges twice.
    // What we really want is an Eulerian cycle for the complete digraph over all possible
    // `SeekTarget`s, but it's not worth spending the time to compute it.
    for from in every_target() {
        seek_to_target(from);

        for to in every_target() {
            seek_to_target(to);
            seek_to_target(from);
        }
    }
}

const BASIC_BLOCK_OFFSET: usize = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeekTarget {
    BlockStart(BasicBlock),
    Before(Location),
    After(Location),
    AfterAssumeCallReturns(Location),
}

impl SeekTarget {
    fn block(&self) -> BasicBlock {
        use SeekTarget::*;

        match *self {
            BlockStart(block) => block,
            Before(loc) | After(loc) | AfterAssumeCallReturns(loc) => loc.block,
        }
    }

    /// Returns the largest index (not including the basic block index) that should be set when
    /// seeking to this location.
    ///
    /// The index for `AfterAssumeCallReturns` is the same as `After` unless the cursor is pointing
    /// at a `Call` terminator that can return. Returns `None` for `BlockStart`.
    fn effect(&self, body: &mir::Body<'_>) -> Option<usize> {
        use SeekTarget::*;

        let idx = match *self {
            BlockStart(_) => return None,

            AfterAssumeCallReturns(loc) if is_call_terminator_non_diverging(body, loc)
                => loc.statement_index * 2 + 2,

            Before(loc) => loc.statement_index * 2,
            After(loc) | AfterAssumeCallReturns(loc) => loc.statement_index * 2 + 1,
        };

        assert!(idx < BASIC_BLOCK_OFFSET, "Too many statements in basic block");
        Some(idx)
    }

    /// An iterator over all possible `SeekTarget`s in a given block in order, starting with
    /// `BlockStart`.
    ///
    /// This includes both `After` and `AfterAssumeCallReturns` for every `Location`.
    fn iter_in_block(body: &mir::Body<'_>, block: BasicBlock) -> impl Iterator<Item = Self> {
        let statements_and_terminator = (0..=body[block].statements.len())
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(move |(i, kind)| {
                let loc = Location { block, statement_index: i };
                match kind {
                    0 => SeekTarget::Before(loc),
                    1 => SeekTarget::After(loc),
                    2 => SeekTarget::AfterAssumeCallReturns(loc),
                    _ => unreachable!(),
                }
            });

        std::iter::once(SeekTarget::BlockStart(block))
            .chain(statements_and_terminator)
    }
}

/// A mock dataflow analysis for testing `ResultsCursor`.
///
/// `Dummy` assigns a unique, monotonically increasing index to each possible cursor position as
/// well as one to each basic block. Instead of being iterated to fixpoint, `Dummy::entry_sets` is
/// used to construct the entry set to each block such that the dataflow state that should be
/// observed by `ResultsCursor` is unique for every location (see `expected_state_at`).
struct Dummy<'tcx> {
    body: &'tcx mir::Body<'tcx>,
}

impl Dummy<'tcx> {
    fn entry_sets(&self) -> IndexVec<BasicBlock, BitSet<usize>> {
        let empty = BitSet::new_empty(self.bits_per_block(self.body));
        let mut ret = IndexVec::from_elem(empty, &self.body.basic_blocks());

        for (bb, _) in self.body.basic_blocks().iter_enumerated() {
            ret[bb].insert(BASIC_BLOCK_OFFSET + bb.index());
        }

        ret
    }

    /// Returns the expected state at the given `SeekTarget`.
    ///
    /// This is the union of index of the target basic block, the index assigned to the of the
    /// target statement or terminator, and the indices of all preceding statements in the MIR.
    ///
    /// For example, the expected state when calling
    /// `seek_before(Location { block: 2, statement_index: 2 })` would be `[102, 0, 1, 2, 3, 4]`.
    fn expected_state_at(&self, target: SeekTarget) -> BitSet<usize> {
        let mut ret = BitSet::new_empty(self.bits_per_block(self.body));
        ret.insert(BASIC_BLOCK_OFFSET + target.block().index());

        if let Some(target_effect) = target.effect(self.body) {
            for i in 0..=target_effect {
                ret.insert(i);
            }
        }

        ret
    }
}

impl BottomValue for Dummy<'tcx> {
    const BOTTOM_VALUE: bool = false;
}

impl AnalysisDomain<'tcx> for Dummy<'tcx> {
    type Idx = usize;

    const NAME: &'static str = "dummy";

    fn bits_per_block(&self, body: &mir::Body<'tcx>) -> usize {
        BASIC_BLOCK_OFFSET + body.basic_blocks().len()
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut BitSet<Self::Idx>) {
        unimplemented!();
    }
}

impl Analysis<'tcx> for Dummy<'tcx> {
    fn apply_statement_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        let idx = SeekTarget::After(location).effect(self.body).unwrap();
        assert!(state.insert(idx));
    }

    fn apply_before_statement_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        let idx = SeekTarget::Before(location).effect(self.body).unwrap();
        assert!(state.insert(idx));
    }

    fn apply_terminator_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        let idx = SeekTarget::After(location).effect(self.body).unwrap();
        assert!(state.insert(idx));
    }

    fn apply_before_terminator_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        let idx = SeekTarget::Before(location).effect(self.body).unwrap();
        assert!(state.insert(idx));
    }

    fn apply_call_return_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        block: BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        _return_place: &mir::Place<'tcx>,
    ) {
        let location = self.body.terminator_loc(block);
        let idx = SeekTarget::AfterAssumeCallReturns(location).effect(self.body).unwrap();
        assert!(state.insert(idx));
    }
}

fn is_call_terminator_non_diverging(body: &mir::Body<'_>, loc: Location) -> bool {
    loc == body.terminator_loc(loc.block)
        && matches!(
            body[loc.block].terminator().kind,
            mir::TerminatorKind::Call { destination: Some(_), ..  }
        )
}

fn dummy_body() -> mir::Body<'static> {
    let span = syntax_pos::DUMMY_SP;
    let source_info = mir::SourceInfo { scope: mir::OUTERMOST_SOURCE_SCOPE, span };

    let mut blocks = IndexVec::new();
    let mut block = |n, kind| {
        let nop = mir::Statement {
            source_info,
            kind: mir::StatementKind::Nop,
        };

        blocks.push(mir::BasicBlockData {
            statements: std::iter::repeat(&nop).cloned().take(n).collect(),
            terminator: Some(mir::Terminator { source_info, kind }),
            is_cleanup: false,
        })
    };

    let dummy_place = mir::Place {
        base: mir::PlaceBase::Local(mir::RETURN_PLACE),
        projection: ty::List::empty(),
    };

    block(4, mir::TerminatorKind::Return);
    block(1, mir::TerminatorKind::Return);
    block(2, mir::TerminatorKind::Call {
        func: mir::Operand::Copy(dummy_place.clone()),
        args: vec![],
        destination: Some((dummy_place.clone(), mir::START_BLOCK)),
        cleanup: None,
        from_hir_call: false,
    });
    block(3, mir::TerminatorKind::Return);
    block(4, mir::TerminatorKind::Call {
        func: mir::Operand::Copy(dummy_place.clone()),
        args: vec![],
        destination: Some((dummy_place.clone(), mir::START_BLOCK)),
        cleanup: None,
        from_hir_call: false,
    });

    mir::Body::new_cfg_only(blocks)
}
