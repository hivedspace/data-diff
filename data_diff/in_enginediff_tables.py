import logging
from functools import partial

from .joindiff_tables import JoinDiffer, bool_to_int
from .sqeleton import this
from .sqeleton.queries import sum_, and_
from .sqeleton.queries.api import not_, or_
from .sqeleton.queries.ast_classes import Cast, Code
from .sqeleton.utils import safezip
from .table_segment import TableSegment
from .diff_tables import DiffResultWrapper, DiffStats, DiffResult
from .info_tree import InfoTree, SegmentInfo
from .thread_utils import ThreadedYielder

logger = logging.getLogger("inenginediff_tables")


class InEngineDiffResultWrapper(DiffResultWrapper):
    def _get_stats(self, is_dbt: bool = False) -> DiffStats:
        diff_by_sign = self.stats["diff_by_sign"]
        table1_count = self.info_tree.info.rowcounts[1]
        table2_count = self.info_tree.info.rowcounts[2]
        unchanged = table1_count - diff_by_sign["-"] - diff_by_sign["!"]
        diff_percent = 1 - unchanged / max(table1_count, table2_count)

        extra_columns = self.info_tree.info.tables[0].extra_columns
        extra_column_diffs = {k: 0 for k in sorted(extra_columns)}
        for key in extra_column_diffs.keys():
            extra_column_diffs[key] = self.stats["full_stats"][f"{key}_a"]

        return DiffStats(diff_by_sign, table1_count, table2_count, unchanged, diff_percent, extra_column_diffs)


class InEngineJoinDiffer(JoinDiffer):
    def diff_tables(self, table1: TableSegment, table2: TableSegment, info_tree: InfoTree = None) -> DiffResultWrapper:
        if info_tree is None:
            info_tree = InfoTree(SegmentInfo([table1, table2]))
        return InEngineDiffResultWrapper(self._diff_tables_wrapper(table1, table2, info_tree), info_tree, self.stats)

    def _get_updated_cols(self, db, diff_rows, cols, is_diff_cols):
        logger.debug("Counting differences per column")
        is_diff_cols_counts = db.query(
            diff_rows.select(
                sum_(
                    bool_to_int(
                        and_(
                            Cast(this[c], Code("BOOL")),
                            not_(this.is_exclusive_a),
                            not_(this.is_exclusive_b),
                        )
                    )
                )
                for c in is_diff_cols
            ),
            tuple,
        )
        diff_counts = {}
        for name, count in safezip(cols, is_diff_cols_counts):
            diff_counts[name] = diff_counts.get(name, 0) + (count or 0)
        self.stats["full_stats"] = diff_counts

    def _get_rows_summary(self, db, diff_rows, is_diff_cols):
        logger.debug("Counting rows on differences")
        query_results = db.query(
            diff_rows.select(
                sum_(bool_to_int(this.is_exclusive_a)),
                sum_(bool_to_int(this.is_exclusive_b)),
                sum_(
                    bool_to_int(
                        or_(
                            and_(
                                Cast(this[c], Code("BOOL")),
                                not_(this.is_exclusive_a),
                                not_(this.is_exclusive_b),
                            )
                            for c in is_diff_cols
                        )
                    )
                ),
            ),
            tuple,
        )
        diff_counts = {}
        for name, count in safezip("-+!", query_results):
            diff_counts[name] = diff_counts.get(name, 0) + (count or 0)
        self.stats["diff_by_sign"] = diff_counts

    def _diff_segments(
        self,
        ti: ThreadedYielder,
        table1: TableSegment,
        table2: TableSegment,
        info_tree: InfoTree,
        max_rows: int,
        level=0,
        segment_index=None,
        segment_count=None,
    ) -> DiffResult:
        assert table1.database is table2.database

        if segment_index or table1.min_key or max_rows:
            logger.info(
                ". " * level + f"Diffing segment {segment_index}/{segment_count}, "
                f"key-range: {table1.min_key}..{table2.max_key}, "
                f"size <= {max_rows}"
            )

        db = table1.database
        diff_rows, a_cols, b_cols, is_diff_cols, all_rows = self._create_outer_join(table1, table2)

        with self._run_in_background(
            partial(self._collect_stats, 1, table1, info_tree),
            partial(self._collect_stats, 2, table2, info_tree),
            partial(self._test_null_keys, table1, table2),
            partial(self._sample_and_count_exclusive, db, diff_rows, a_cols, b_cols),
            partial(self._count_diff_per_column, db, diff_rows, list(a_cols), is_diff_cols),
            partial(
                self._materialize_diff,
                db,
                all_rows if self.materialize_all_rows else diff_rows,
                segment_index=segment_index,
            )
            if self.materialize_to_table
            else None,
        ):
            assert len(a_cols) == len(b_cols)
            self._get_updated_cols(db, diff_rows, list(a_cols), is_diff_cols)
            self._get_rows_summary(db, diff_rows, is_diff_cols)
            yield "done", tuple()
