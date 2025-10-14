"""Population module.
A population is a collection of portfolios.
"""

# failure_proba
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.interpolate as sci

import skfolio.typing as skt
from skfolio.measures import RatioMeasure
from skfolio.portfolio import BasePortfolio, FailedPortfolio, MultiPeriodPortfolio
from skfolio.utils.figure import kde_trace
from skfolio.utils.sorting import non_denominated_sort
from skfolio.utils.tools import deduplicate_names, optimal_rounding_decimals


class Population(list):
    """Population Class.

    A `Population` is a list of :class:`~skfolio.portfolio.Portfolio` or
    :class:`~skfolio.portfolio.MultiPeriodPortfolio` or both.

    Parameters
    ----------
    iterable : list[BasePortfolio]
        The list of portfolios. Each item can be of type
        :class:`~skfolio.portfolio.Portfolio` and/or
        :class:`~skfolio.portfolio.MultiPeriodPortfolio`.
        Empty list are accepted.
    """

    def __init__(self, iterable: list[BasePortfolio]) -> None:
        super().__init__(self._validate_item(item) for item in iterable)

    def __repr__(self) -> str:
        return "<Population(" + super().__repr__() + ")>"

    def __getitem__(
        self, indices: int | list[int] | slice
    ) -> "BasePortfolio | Population":
        item = super().__getitem__(indices)
        if isinstance(item, list):
            return self.__class__(item)
        return item

    def __setitem__(self, index: int, item: BasePortfolio) -> None:
        super().__setitem__(index, self._validate_item(item))

    def __add__(self, other: BasePortfolio) -> "Population":
        if not isinstance(other, Population):
            raise TypeError(
                f"Cannot add a Population with an object of type {type(other)}"
            )
        return self.__class__(super().__add__(other))

    def insert(self, index, item: BasePortfolio) -> None:
        """Insert portfolio before index."""
        super().insert(index, self._validate_item(item))

    def append(self, item: BasePortfolio) -> None:
        """Append portfolio to the end of the population list."""
        super().append(self._validate_item(item))

    def extend(self, other: BasePortfolio) -> None:
        """Extend population list by appending elements from the iterable."""
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(self._validate_item(item) for item in other)

    def set_portfolio_params(self, **params: Any) -> "Population":
        """Set the parameters of all the portfolios.

        Parameters
        ----------
        **params : Any
            Portfolio parameters.

        Returns
        -------
        self : Population
            The Population instance.
        """
        if not params:
            return self
        init_signature = inspect.signature(BasePortfolio.__init__)
        # Consider the constructor parameters excluding 'self'
        valid_params = [
            p.name
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for key in params:
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} . "
                    f"Valid parameters are: {valid_params!r}."
                )

        for portfolio in self:
            for key, value in params.items():
                setattr(portfolio, key, value)

    @staticmethod
    def _validate_item(
        item: BasePortfolio,
    ) -> BasePortfolio:
        """Validate that items are of type Portfolio or MultiPeriodPortfolio."""
        if isinstance(item, BasePortfolio):
            return item
        raise TypeError(
            "Population only accept items that inherit from BasePortfolio such as "
            "Portfolio or MultiPeriodPortfolio"
            f", got {type(item).__name__}"
        )

    def _validate_compounded(self) -> bool:
        """
        Determine whether all portfolios in the population use compounded returns.

        Returns
        -------
        bool
            True if all portfolios are compounded, False if all are non-compounded.

        Raises
        ------
        ValueError
            If the population is empty, or if it mixes compounded and non-compounded
            portfolios.
        """
        compounded = [ptf.compounded for ptf in self]

        if not compounded:
            raise ValueError("Cannot determine compounded status: population is empty.")

        compounded = set(compounded)
        if len(compounded) > 1:
            raise ValueError(
                "Population contains a mix of compounded and non-compounded portfolios."
                " Ensure consistency, for example with "
                "`population.set_portfolio_params(compounded=False)`."
            )

        return compounded.pop()

    def cumulative_returns_df(
        self, use_tag_in_column_name: bool = True
    ) -> pd.DataFrame:
        """DataFrame of cumulative returns for each portfolio in the population.
        Non-compounded (arithmetic) cumulative returns start at 0.
        Compounded (geometric) cumulative returns are expressed as a wealth index,
        starting at 1.0 (i.e., the value of $1 invested).

        Parameters
        ----------
        use_tag_in_column_name : bool, default=True
            Whether to include the portfolio tag in the DataFrame column names.
            If True, each column name will use the portfolio name followed by its tag;
            if False, only the portfolio name will be used.

        Returns
        -------
        cumulative_returns : DataFrame
            Cumulative returns DataFrame.
        """
        self._validate_compounded()
        cumulative_returns = []
        names = []
        for ptf in self:
            cumulative_returns.append(ptf.cumulative_returns_df)
            names.append(
                _ptf_name_with_tag(ptf) if use_tag_in_column_name else ptf.name
            )
        df = pd.concat(cumulative_returns, axis=1)
        # Sort index because pd.concat unsort NaNs at the end
        df.sort_index(inplace=True)
        df.columns = deduplicate_names(names)
        return df

    def drawdowns_df(self, use_tag_in_column_name: bool = True) -> pd.DataFrame:
        """DataFrame of drawdowns for each portfolio in the population.

        Parameters
        ----------
        use_tag_in_column_name : bool, default=True
            Whether to include the portfolio tag in the DataFrame column names.
            If True, each column name will use the portfolio name followed by its tag;
            if False, only the portfolio name will be used.

        Returns
        -------
        drawdowns : DataFrame
            Drawdowns DataFrame.
        """
        self._validate_compounded()
        drawdowns = []
        names = []
        for ptf in self:
            drawdowns.append(ptf.drawdowns_df)
            names.append(
                _ptf_name_with_tag(ptf) if use_tag_in_column_name else ptf.name
            )
        df = pd.concat(drawdowns, axis=1)
        # Sort index because pd.concat unsort NaNs at the end
        df.sort_index(inplace=True)
        df.columns = deduplicate_names(names)
        return df

    def non_denominated_sort(self, first_front_only: bool = False) -> list[list[int]]:
        """Fast non-dominated sorting.
        Sort the portfolios into different non-domination levels.
        Complexity O(MN^2) where M is the number of objectives and N the number of
        portfolios.

        Parameters
        ----------
        first_front_only : bool, default=False
            If this is set to True, only the first front is sorted and returned.
            The default is `False`.

        Returns
        -------
        fronts : list[list[int]]
            A list of Pareto fronts (lists), the first list includes
            non-dominated portfolios.
        """
        n = len(self)
        if n > 0 and np.any(
            [
                portfolio.fitness_measures != self[0].fitness_measures
                for portfolio in self
            ]
        ):
            raise ValueError(
                "Cannot compute non denominated sorting with Portfolios "
                "containing mixed `fitness_measures`"
            )
        fitnesses = np.array([portfolio.fitness for portfolio in self])
        fronts = non_denominated_sort(
            fitnesses=fitnesses, first_front_only=first_front_only
        )
        return fronts

    def filter(
        self, names: skt.Names | None = None, tags: skt.Tags | None = None
    ) -> "Population":
        """Filter the Population of portfolios by names and tags.
        If both names and tags are provided, the intersection is returned.

        Parameters
        ----------
        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags :  str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        population : Population
            A new population of portfolios filtered by names and tags.
        """
        if tags is None and names is None:
            return self
        if isinstance(names, str):
            names = [names]
        if isinstance(tags, str):
            tags = [tags]

        if tags is None:
            return self.__class__(
                [portfolio for portfolio in self if portfolio.name in names]
            )
        if names is None:
            return self.__class__(
                [portfolio for portfolio in self if portfolio.tag in tags]
            )
        return self.__class__(
            [
                portfolio
                for portfolio in self
                if portfolio.name in names and portfolio.tag in tags
            ]
        )

    def measures(
        self,
        measure: skt.Measure,
    ) -> np.ndarray:
        """Vector of portfolios measures for each portfolio from the
        population.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        Returns
        -------
        values : ndarray
            The vector of portfolios measures.
        """
        return np.array([ptf.__getattribute__(measure.value) for ptf in self])

    def measures_mean(
        self,
        measure: skt.Measure,
    ) -> float:
        """Mean of portfolios measures for each portfolio from the
        population.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        Returns
        -------
        value : float
            The mean of portfolios measures.
        """
        return np.nanmean(self.measures(measure=measure), axis=0)

    def measures_std(
        self,
        measure: skt.Measure,
    ) -> float:
        """Standard-deviation of portfolios measures for each portfolio from the
        population.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        Returns
        -------
        value : float
            The standard-deviation of portfolios measures.
        """
        return np.nanstd(self.measures(measure=measure), axis=0)

    def sort_measure(self, measure: skt.Measure, reverse: bool = False) -> "Population":
        """Sort the population by a given portfolio measure.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        reverse : bool, default=False
            If this is set to True, the order is reversed.

        Returns
        -------
        values : Populations
            The sorted population.
        """
        return self.__class__(
            sorted(
                [x for x in self if not isinstance(x, FailedPortfolio)],
                key=lambda x: x.__getattribute__(measure.value),
                reverse=reverse,
            )
        )

    def quantile(
        self,
        measure: skt.Measure,
        q: float,
    ) -> BasePortfolio:
        """Return the portfolio corresponding to the `q` quantile for a given portfolio
        measure.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        q : float
            The quantile value.

        Returns
        -------
        values : BasePortfolio
           Portfolio corresponding to the `q` quantile for the measure.
        """
        if not 0 <= q <= 1:
            raise ValueError("The quantile`q` must be between 0 and 1")
        sorted_portfolios = self.sort_measure(measure=measure, reverse=False)
        k = max(0, int(np.round(len(sorted_portfolios) * q)) - 1)
        return sorted_portfolios[k]

    def min_measure(
        self,
        measure: skt.Measure,
    ) -> BasePortfolio:
        """Return the portfolio with the minimum measure.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        Returns
        -------
        values : BasePortfolio
            The portfolio with minimum measure.
        """
        return self.quantile(measure=measure, q=0)

    def max_measure(
        self,
        measure: skt.Measure,
    ) -> BasePortfolio:
        """Return the portfolio with the maximum measure.

        Parameters
        ----------
        measure: Measure
            The portfolio measure.

        Returns
        -------
        values : BasePortfolio
            The portfolio with maximum measure.
        """
        return self.quantile(measure=measure, q=1)

    def summary(
        self,
        formatted: bool = True,
    ) -> pd.DataFrame:
        """Summary of the portfolios in the population.

        Parameters
        ----------
        formatted : bool, default=True
            If this is set to True, the measures are formatted into rounded string with
            units.
            The default is `True`.

        Returns
        -------
        summary : pandas DataFrame
            The population's portfolios summary

        Notes
        -----
        This method returns a static pandas DataFrame. For interactive exploration
        (e.g., sortable/filterable/clickable tables or visual summaries), you may want
        to use libraries such as `ipydatagrid`, `D-Tale`, or `Lux` in a Jupyter
        environment, or `dash_table` / `streamlit.dataframe` when building dashboards.
        For example, you can explore the summary with D-Tale:
        `dtale.show(population.summary().T)`
        """
        df = pd.concat(
            [p.summary(formatted=formatted) for p in self],
            keys=[p.name for p in self],
            axis=1,
        )
        return df

    def composition(
        self,
        display_sub_ptf_name: bool = True,
    ) -> pd.DataFrame:
        """Composition of each portfolio in the population.

        Parameters
        ----------
        display_sub_ptf_name : bool, default=True
            If this is set to True, each sub-portfolio name composing a multi-period
            portfolio is displayed.

        Returns
        -------
        df : DataFrame
            Composition of the portfolios in the population.
        """
        res = []
        for ptf in self:
            comp = ptf.composition
            if display_sub_ptf_name:
                if isinstance(ptf, MultiPeriodPortfolio):
                    comp.rename(
                        columns={c: f"{ptf.name}_{c}" for c in comp.columns},
                        inplace=True,
                    )
            else:
                comp.rename(columns={c: ptf.name for c in comp.columns}, inplace=True)
            res.append(comp)

        df = pd.concat(res, axis=1)
        # Leave columns of only NaNs untouched
        mask = ~df.isna().all(axis=0)
        df.loc[:, mask] = df.loc[:, mask].fillna(0)
        df.columns = deduplicate_names(list(df.columns))
        return df

    def contribution(
        self,
        measure: skt.Measure,
        spacing: float | None = None,
        display_sub_ptf_name: bool = True,
    ) -> pd.DataFrame:
        r"""Contribution of each asset to a given measure of each portfolio in the
        population.

        Parameters
        ----------
        measure : Measure
            The measure used for the contribution computation.

        spacing : float, optional
            Spacing "h" of the finite difference:
            :math:`contribution(wi)= \frac{measure(wi-h) - measure(wi+h)}{2h}`.

        display_sub_ptf_name : bool, default=True
            If this is set to True, each sub-portfolio name composing a multi-period
            portfolio is displayed.

        Returns
        -------
        df : DataFrame
            Contribution of each asset to a given measure of each portfolio in the
            population.
        """
        res = []
        for ptf in self:
            contribution = ptf.contribution(
                measure=measure, spacing=spacing, to_df=True
            )
            if display_sub_ptf_name:
                if isinstance(ptf, MultiPeriodPortfolio):
                    contribution.rename(
                        columns={c: f"{ptf.name}_{c}" for c in contribution.columns},
                        inplace=True,
                    )
            else:
                contribution.rename(
                    columns={c: ptf.name for c in contribution.columns}, inplace=True
                )
            res.append(contribution)

        df = pd.concat(res, axis=1)
        # Leave columns of only NaNs untouched
        mask = ~df.isna().all(axis=0)
        df.loc[:, mask] = df.loc[:, mask].fillna(0)
        df.columns = deduplicate_names(list(df.columns))
        return df

    def rolling_measure(
        self, measure: skt.Measure = RatioMeasure.SHARPE_RATIO, window: int = 30
    ) -> pd.DataFrame:
        """Compute the measure over a rolling window for each portfolio in the
         population.

        Parameters
        ----------
        measure : ct.Measure, default=RatioMeasure.SHARPE_RATIO
            The measure. The default measure is the Sharpe Ratio.

        window : int, default=30
            The window size. The default value is `30` observations.

        Returns
        -------
        dataframe : pandas DataFrame
            The rolling measures.
        """
        rolling_measures = []
        names = []
        for ptf in self:
            rolling_measures.append(ptf.rolling_measure(measure=measure, window=window))
            names.append(_ptf_name_with_tag(ptf))
        df = pd.concat(rolling_measures, axis=1)
        df.columns = deduplicate_names(names)
        # Sort index because pd.concat unsort NaNs at the end
        df.sort_index(inplace=True)
        return df

    def plot_distribution(
        self,
        measure_list: list[skt.Measure],
        tag_list: list[str] | None = None,
        n_bins: int | None = None,
        **kwargs,
    ) -> go.Figure:
        """Plot the population's distribution for each measure provided in the
        measure list.

        Parameters
        ----------
        measure_list : list[Measure]
            The list of portfolio measures. A different distribution is plotted per
            measure.

        tag_list : list[str], optional
            If this is provided, an additional distribution is plotted per measure
            for each tag provided.

        n_bins : int, optional
            Sets the number of bins.

        Returns
        -------
        plot : Figure
            Returns the plotly Figure object.
        """
        n = len(measure_list)

        if tag_list is None:
            df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "Population": measure.value,
                            "value": self.measures(measure=measure),
                        }
                    )
                    for measure in measure_list
                ],
                ignore_index=True,
            )
        else:
            df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "Population": tag if n == 1 else f"{measure} - {tag}",
                            "value": self.filter(tags=tag).measures(measure=measure),
                        }
                    )
                    for measure in measure_list
                    for tag in tag_list
                ],
                ignore_index=True,
            )

        fig = px.histogram(
            df,
            color="Population",
            barmode="overlay",
            marginal="box",
            nbins=n_bins,
            **kwargs,
        )
        title = f"{measure_list[0]} Distribution" if n == 1 else "Measures Distribution"
        fig.update_layout(
            title_text=title, xaxis_title=str(measure_list[0]) if n == 1 else "measures"
        )
        return fig

    def boxplot_measure(
        self,
        measure: skt.Measure,
        tag_list: list[str] | None = None,
        points: str | bool = "all",
    ) -> go.Figure:
        """Plot a box plot of a measure's distribution, optionally split by tags.

        If no tags are provided, the function draws a single box showing the
        population distribution of `measure`. If `tag_list` is provided, it draws
        one box per tag using values from the portfolio filtered by each tag.

        Parameters
        ----------
        measure : Measure
            The measure to plot.

        tag_list : list[str], optional
            For each tag in this list, filter the portfolio by that tag and plot a
            separate box. If None or empty, plot a single overall distribution.

        points : {'all', 'outliers', 'suspectedoutliers', False}, default 'all'
            Passed to `plotly.express.box(..., points=...)` to control which points
            are shown.

        Returns
        -------
        go.Figure
            The Plotly figure.

        Examples
        --------
        >>> fig = population.boxplot_measure(measure=RiskMeasure.STANDARD_DEVIATION)
        >>> fig = population.plot_measure_box(
        ...     measure=RatioMeasure.SHARPE_RATIO,
        ...     tag_list=["Benchmark", "Risk Parity Model"]
        ... )
        """
        if tag_list is None:
            y = None
            df = pd.DataFrame(self.measures(measure=measure), columns=["value"])
        else:
            y = "Population"
            df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            y: tag,
                            "value": self.filter(tags=tag).measures(measure=measure),
                        }
                    )
                    for tag in tag_list
                ],
                ignore_index=True,
            )

        fig = px.box(df, x="value", y=y, color=y, points=points)
        fig.update_layout(title_text=f"Box plot of {measure}", xaxis_title=str(measure))
        return fig

    def plot_cumulative_returns(
        self,
        log_scale: bool = False,
        idx: slice | np.ndarray | None = None,
        use_tag_in_legend: bool = True,
    ) -> go.Figure:
        """Plot the cumulative returns of the population's portfolios.
        Non-compounded (arithmetic) cumulative returns start at 0.
        Compounded (geometric) cumulative returns are expressed as a wealth index,
        starting at 1.0 (i.e., the value of $1 invested).

        Parameters
        ----------
        log_scale : bool, default=False
            If set to True, the cumulative returns are displayed with a
            logarithm scale on the y-axis. The cumulative returns must be compounded
            otherwise an exception is raise.

        idx : slice | array, optional
            Indexes or slice of the observations to plot.
            The default (`None`) is to take all observations.

        use_tag_in_legend : bool, default=True
            Whether to include the portfolio tag in legend entries.
            If True, each legend label will show the portfolio name followed by its tag;
            if False, only the portfolio name will be displayed.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object.
        """
        if idx is None:
            idx = slice(None)

        compounded = self._validate_compounded()
        title = "Cumulative Returns"
        if compounded:
            if log_scale:
                title = f"{title} (compounded & log scaled)"
            else:
                title = f"{title} (compounded)"
        else:
            if log_scale:
                raise ValueError(
                    "Plotting with logarithm scaling must be done on cumulative "
                    "returns that are compounded as opposed to non-compounded."
                    "You can change to compounded with "
                    "`set_portfolio_params(compounded=True)`"
                )
            title = f"{title} (non-compounded)"

        df = self.cumulative_returns_df(use_tag_in_column_name=use_tag_in_legend)
        fig = df.iloc[idx].plot(backend="plotly")
        fig.update_layout(
            title=title,
            xaxis_title="Observations",
            yaxis_title="Cumulative Returns",
            legend_title_text="Portfolios",
        )
        if compounded:
            fig.update_yaxes(tickformat=".2f")
        else:
            fig.update_yaxes(tickformat=".2%")
        if log_scale:
            fig.update_yaxes(type="log")
        return fig

    def plot_drawdowns(
        self,
        idx: slice | np.ndarray | None = None,
        use_tag_in_legend: bool = True,
    ) -> go.Figure:
        """Plot the drawdowns of the population's portfolios.

        Parameters
        ----------
        idx : slice | array, optional
            Indexes or slice of the observations to plot.
            The default (`None`) is to take all observations.

        use_tag_in_legend : bool, default=True
            Whether to include the portfolio tag in legend entries.
            If True, each legend label will show the portfolio name followed by its tag;
            if False, only the portfolio name will be displayed.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object.
        """
        if idx is None:
            idx = slice(None)

        compounded = self._validate_compounded()
        title = "Drawdowns"
        if compounded:
            title = f"{title} (compounded returns)"
        else:
            title = f"{title} (non-compounded returns)"

        df = self.drawdowns_df(use_tag_in_column_name=use_tag_in_legend)
        fig = df.iloc[idx].plot(backend="plotly")
        fig.update_layout(
            title=title,
            xaxis_title="Observations",
            yaxis_title="Drawdowns",
            legend_title_text="Portfolios",
        )
        fig.update_yaxes(tickformat=".1%")
        return fig

    def plot_composition(self, display_sub_ptf_name: bool = True) -> go.Figure:
        """Plot the compositions of the portfolios in the population.

        Parameters
        ----------
        display_sub_ptf_name : bool, default=True
            If this is set to True, each sub-portfolio name composing a multi-period
            portfolio is displayed.

        Returns
        -------
        plot : Figure
            Returns the plotly Figure object.
        """
        df = self.composition(display_sub_ptf_name=display_sub_ptf_name).T
        fig = px.bar(df, x=df.index, y=df.columns)
        fig.update_layout(
            title="Portfolios Composition",
            xaxis_title="Portfolios",
            yaxis={
                "title": "Weight",
                "tickformat": ",.0%",
            },
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.15),
        )
        return fig

    def plot_contribution(
        self,
        measure: skt.Measure,
        spacing: float | None = None,
        display_sub_ptf_name: bool = True,
    ) -> go.Figure:
        r"""Plot the contribution of each asset to a given measure of the portfolios
        in the population.

        Parameters
        ----------
        measure : Measure
            The measure used for the contribution computation.

        spacing : float, optional
            Spacing "h" of the finite difference:
            :math:`contribution(wi)= \frac{measure(wi-h) - measure(wi+h)}{2h}`

        display_sub_ptf_name : bool, default=True
            If this is set to True, each sub-portfolio name composing a multi-period
            portfolio is displayed.

        Returns
        -------
        plot : Figure
            Returns the plotly Figure object.
        """
        df = self.contribution(
            display_sub_ptf_name=display_sub_ptf_name, measure=measure, spacing=spacing
        ).T
        fig = px.bar(df, x=df.index, y=df.columns)

        yaxis = {
            "title": "Contribution",
        }
        if not measure.is_ratio:
            n = optimal_rounding_decimals(df.sum(axis=1).max())
            yaxis["tickformat"] = f",.{n}%"

        fig.update_layout(
            title=f"{measure} Contribution",
            xaxis_title="Portfolios",
            yaxis=yaxis,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.15),
        )
        return fig

    def plot_measures(
        self,
        x: skt.Measure,
        y: skt.Measure,
        z: skt.Measure = None,
        to_surface: bool = False,
        hover_measures: list[skt.Measure] | None = None,
        show_fronts: bool = False,
        color_scale: skt.Measure | str | None = None,
        title="Portfolios",
    ) -> go.Figure:
        """Plot the 2D (or 3D) scatter points (or surface) of a given set of
        measures for each portfolio in the population.

        Parameters
        ----------
        x : Measure
            The x-axis measure.

        y : Measure
            The y-axis measure.

        z : Measure, optional
            The z-axis measure.

        to_surface : bool, default=False
            If this is set to True, a surface is estimated.

        hover_measures : list[Measure], optional
            The list of measure to show on point hover.

        show_fronts : bool, default=False
            If this is set to True, the pareto fronts are highlighted.
            The default is `False`.

        color_scale : Measure | str, optional
            If this is provided, a color scale is displayed.

        title : str, default="Portfolios"
            The graph title. The default value is "Portfolios".

        Returns
        -------
        plot : Figure
            Returns the plotly Figure object.
        """
        num_fmt = ":.3f"
        hover_data = {x: num_fmt, y: num_fmt, "tag": True}

        if z is not None:
            hover_data[z] = num_fmt

        if hover_measures is not None:
            for measure in hover_measures:
                hover_data[measure] = num_fmt

        columns = list(hover_data)
        columns.append("name")
        if isinstance(color_scale, skt.Measure):
            hover_data[color_scale] = num_fmt

        if color_scale is not None and color_scale not in columns:
            columns.append(color_scale)

        col_values = [e.value if isinstance(e, skt.Measure) else e for e in columns]
        res = [
            [portfolio.__getattribute__(attr) for attr in col_values]
            for portfolio in self
        ]
        # Improved formatting
        columns = [str(e) for e in columns]
        hover_data = {str(k): v for k, v in hover_data.items()}

        df = pd.DataFrame(res, columns=columns)
        if pd.isnull(df["tag"]).all():
            del hover_data["tag"]
            tag = None
        else:
            tag = "tag"
            df["tag"] = df["tag"].astype(str).replace("None", "")

        if show_fronts:
            fronts = self.non_denominated_sort(first_front_only=False)
            df["front"] = str(-1)
            for i, front in enumerate(fronts):
                for idx in front:
                    df.iloc[idx, -1] = str(i)
            color = df.columns[-1]
        elif color_scale is not None:
            color = str(color_scale)
        else:
            color = tag

        if z is not None:
            if to_surface:
                # estimate the surface
                x_arr = np.array(df[str(x)])
                y_arr = np.array(df[str(y)])
                z_arr = np.array(df[str(z)])

                xi = np.linspace(start=min(x_arr), stop=max(x_arr), num=100)
                yi = np.linspace(start=min(y_arr), stop=max(y_arr), num=100)

                X, Y = np.meshgrid(xi, yi)
                Z = sci.griddata(
                    points=(x_arr, y_arr), values=z_arr, xi=(X, Y), method="cubic"
                )
                fig = go.Figure(
                    go.Surface(
                        x=xi,
                        y=yi,
                        z=Z,
                        hovertemplate="<br>".join(
                            [
                                str(e)
                                + ": %{"
                                + v
                                + ":"
                                + (",.3%" if not e.is_ratio else None)
                                + "}"
                                for e, v in [(x, "x"), (y, "y"), (z, "z")]
                            ]
                        )
                        + "<extra></extra>",
                        colorbar=dict(
                            title=dict(text=str(z), side="top"),
                            tickformat=",.2%" if not z.is_ratio else None,
                        ),
                    )
                )

                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis={
                            "title": str(x),
                            "tickformat": ",.1%" if not x.is_ratio else None,
                        },
                        yaxis={
                            "title": str(y),
                            "tickformat": ",.1%" if not y.is_ratio else None,
                        },
                        zaxis={
                            "title": str(z),
                            "tickformat": ",.1%" if not z.is_ratio else None,
                        },
                    ),
                )
            else:
                # plot the points
                fig = px.scatter_3d(
                    df,
                    x=str(x),
                    y=str(y),
                    z=str(z),
                    hover_name="name",
                    hover_data=hover_data,
                    color=color,
                    symbol=tag,
                )
                fig.update_traces(marker_size=8)
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis={
                            "title": str(x),
                            "tickformat": ",.1%" if not x.is_ratio else None,
                        },
                        yaxis={
                            "title": str(y),
                            "tickformat": ",.1%" if not y.is_ratio else None,
                        },
                        zaxis={
                            "title": str(z),
                            "tickformat": ",.1%" if not z.is_ratio else None,
                        },
                    ),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.15),
                )

        else:
            fig = px.scatter(
                df,
                x=str(x),
                y=str(y),
                hover_name="name",
                hover_data=hover_data,
                color=color,
                symbol=tag,
            )
            fig.update_traces(marker_size=10)

            if color_scale is None:
                legend = dict(title=None, yanchor="top", y=0.98, xanchor="left", x=1.02)
            else:
                legend = dict(title=None, yanchor="top", y=0.98, xanchor="left", x=0.02)

            fig.update_layout(
                title=title,
                xaxis={
                    "title": str(x),
                    "tickformat": ",.1%" if not x.is_ratio else None,
                },
                yaxis={
                    "title": str(y),
                    "tickformat": ",.1%" if not y.is_ratio else None,
                },
                legend=legend,
            )
        return fig

    def plot_rolling_measure(
        self,
        measure: skt.Measure = RatioMeasure.SHARPE_RATIO,
        window: int = 30,
    ) -> go.Figure:
        """Plot the measure over a rolling window for each portfolio in the population.

        Parameters
        ----------
        measure : ct.Measure, default = RatioMeasure.SHARPE_RATIO
           The measure.

        window : int, default=30
           The window size.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object
        """
        df = self.rolling_measure(measure=measure, window=window)
        fig = df.plot(backend="plotly")
        max_val = np.max(df)
        min_val = np.min(df)
        if max_val > 0 > min_val:
            fig.add_hrect(
                y0=0, y1=max_val * 1.3, line_width=0, fillcolor="green", opacity=0.1
            )
            fig.add_hrect(
                y0=min_val * 1.3, y1=0, line_width=0, fillcolor="red", opacity=0.1
            )

        yaxis = {
            "title": str(measure),
        }
        if not measure.is_ratio:
            n = optimal_rounding_decimals(max_val)
            yaxis["tickformat"] = f",.{n}%"

        fig.update_layout(
            title=f"Rolling {measure} - {window} observations window",
            xaxis_title="Observations",
            yaxis=yaxis,
            showlegend=False,
        )
        return fig

    def plot_returns_distribution(
        self, percentile_cutoff: float | None = None
    ) -> go.Figure:
        """Plot the Portfolios returns distribution using Gaussian KDE.

        Parameters
        ----------
        percentile_cutoff : float, default=None
            Percentile cutoff for tail truncation (percentile), in percent.
            If a float p is provided, the distribution support is truncated at the p-th
            and (100 - p)-th percentiles.
            If None, no truncation is applied (uses full min/max of returns).

        Returns
        -------
        plot : Figure
            Returns the plot Figure object
        """
        traces: list[go.Scatter] = []
        colors = px.colors.qualitative.Plotly

        for i, ptf in enumerate(self):
            if isinstance(ptf, FailedPortfolio):
                continue
            color = colors[i % len(colors)]
            returns = ptf.returns
            traces.append(
                kde_trace(
                    x=returns,
                    sample_weight=ptf.sample_weight,
                    percentile_cutoff=percentile_cutoff,
                    name=ptf.name,
                    line_color=color,
                    fill_opacity=0.3,
                    line_dash="solid",
                    line_width=1,
                    visible=True,
                )
            )

        fig = go.Figure(traces)
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Returns",
            yaxis_title="Probability Density",
        )
        fig.update_xaxes(
            tickformat=".0%",
        )
        return fig


def _ptf_name_with_tag(portfolio: BasePortfolio) -> str:
    if portfolio.tag is None:
        return portfolio.name
    return f"{portfolio.name}_{portfolio.tag}"
