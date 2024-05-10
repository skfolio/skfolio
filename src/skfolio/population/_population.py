"""Population module.
A population is a collection of portfolios.
"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import inspect
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.interpolate as sci

import skfolio.typing as skt
from skfolio.portfolio import BasePortfolio, MultiPeriodPortfolio
from skfolio.utils.sorting import non_denominated_sort
from skfolio.utils.tools import deduplicate_names

pd.options.plotting.backend = "plotly"


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
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> np.ndarray:
        """Vector of portfolios measures for each portfolio from the
        population filtered by names and tags.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        names :  str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        values : ndarray
            The vector of portfolios measures.
        """
        population = self.filter(names=names, tags=tags)
        return np.array([ptf.__getattribute__(measure.value) for ptf in population])

    def measures_mean(
        self,
        measure: skt.Measure,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> float:
        """Mean of portfolios measures for each portfolio from the
        population filtered by names and tags.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        names :  str | list[str], optional
           If provided, the population is filtered by portfolio names.

        tags :  str | list[str], optional
           If provided, the population is filtered by portfolio tags.

        Returns
        -------
        value : float
            The mean of portfolios measures.
        """
        return self.measures(measure=measure, names=names, tags=tags).mean()

    def measures_std(
        self,
        measure: skt.Measure,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> float:
        """Standard-deviation of portfolios measures for each portfolio from the
        population filtered by names and tags.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        value : float
            The standard-deviation of portfolios measures.
        """
        return self.measures(measure=measure, names=names, tags=tags).std()

    def sort_measure(
        self,
        measure: skt.Measure,
        reverse: bool = False,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> "Population":
        """Sort the population by a given portfolio measure and filter the portfolios
        by names and tags.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        reverse : bool, default=False
            If this is set to True, the order is reversed.

        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        values : Populations
            The sorted population.
        """
        population = self.filter(names=names, tags=tags)
        return self.__class__(
            sorted(
                population,
                key=lambda x: x.__getattribute__(measure.value),
                reverse=reverse,
            )
        )

    def quantile(
        self,
        measure: skt.Measure,
        q: float,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> BasePortfolio:
        """Returns the portfolio corresponding to the `q` quantile for a given portfolio
        measure.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        q : float
            The quantile value.

        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        values : BasePortfolio
           Portfolio corresponding to the `q` quantile for the measure.
        """
        if not 0 <= q <= 1:
            raise ValueError("The quantile`q` must be between 0 and 1")
        sorted_portfolios = self.sort_measure(
            measure=measure, reverse=False, names=names, tags=tags
        )
        k = max(0, int(np.round(len(sorted_portfolios) * q)) - 1)
        return sorted_portfolios[k]

    def min_measure(
        self,
        measure: skt.Measure,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> BasePortfolio:
        """Returns the portfolio with the minimum measure.

        Parameters
        ----------
        measure : Measure
            The portfolio measure.

        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        values : BasePortfolio
            The portfolio with minimum measure.
        """
        return self.quantile(measure=measure, q=0, names=names, tags=tags)

    def max_measure(
        self,
        measure: skt.Measure,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> BasePortfolio:
        """Returns the portfolio with the maximum measure.

        Parameters
        ----------
        measure: Measure
            The portfolio measure.

        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        values : BasePortfolio
            The portfolio with maximum measure.
        """
        return self.quantile(measure=measure, q=1, names=names, tags=tags)

    def summary(
        self,
        formatted: bool = True,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> pd.DataFrame:
        """Summary of the portfolios in the population

        Parameters
        ----------
        formatted : bool, default=True
            If this is set to True, the measures are formatted into rounded string with
            units.
            The default is `True`.

        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        summary : pandas DataFrame
            The population's portfolios summary
        """

        portfolios = self.filter(names=names, tags=tags)
        df = pd.concat(
            [p.summary(formatted=formatted) for p in portfolios],
            keys=[p.name for p in portfolios],
            axis=1,
        )
        return df

    def composition(
        self,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
        display_sub_ptf_name: bool = True,
    ) -> pd.DataFrame:
        """Composition of the portfolios in the population.

        Parameters
        ----------
        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        display_sub_ptf_name : bool, default=True
            If this is set to True, each sub-portfolio name composing a multi-period
            portfolio is displayed.

        Returns
        -------
        summary : DataFrame
            Composition of the portfolios in the population.
        """
        portfolios = self.filter(names=names, tags=tags)
        comp_list = []
        for p in portfolios:
            comp = p.composition
            if display_sub_ptf_name:
                if isinstance(p, MultiPeriodPortfolio):
                    comp.rename(
                        columns={c: f"{p.name}_{c}" for c in comp.columns}, inplace=True
                    )
            else:
                comp.rename(columns={c: p.name for c in comp.columns}, inplace=True)
            comp_list.append(comp)

        df = pd.concat(comp_list, axis=1)
        df.columns = deduplicate_names(list(df.columns))
        df.fillna(0, inplace=True)
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
        values = []
        labels = []
        for measure in measure_list:
            if tag_list is not None:
                for tag in tag_list:
                    values.append(self.measures(measure=measure, tags=tag))
                    labels.append(f"{measure} - {tag}")
            else:
                values.append(self.measures(measure=measure))
                labels.append(measure.value)

        df = pd.DataFrame(np.array(values).T, columns=labels).melt(
            var_name="Population"
        )
        fig = px.histogram(
            df,
            color="Population",
            barmode="overlay",
            marginal="box",
            nbins=n_bins,
            **kwargs,
        )
        fig.update_layout(title_text="Measures Distribution", xaxis_title="measures")
        return fig

    def plot_cumulative_returns(
        self,
        log_scale: bool = False,
        idx: slice | np.ndarray | None = None,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
    ) -> go.Figure:
        """Plot the population's portfolios cumulative returns.
        Non-compounded cumulative returns start at 0.
        Compounded cumulative returns are rescaled to start at 1000.

        Parameters
        ----------
        log_scale : bool, default=False
            If this is set to True, the cumulative returns are displayed with a
            logarithm scale on the y-axis and rebased at 1000. The cumulative returns
            must be compounded otherwise an exception is raise.

        idx : slice | array, optional
            Indexes or slice of the observations to plot.
            The default (`None`) is to take all observations.

        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object.
        """
        if idx is None:
            idx = slice(None)
        portfolios = self.filter(names=names, tags=tags)
        if not portfolios:
            raise ValueError("No portfolio found")

        cumulative_returns = []
        names = []
        compounded = []
        for ptf in portfolios:
            cumulative_returns.append(ptf.cumulative_returns_df)
            names.append(f"{ptf.name}_{ptf.tag}" if ptf.tag is not None else ptf.name)
            compounded.append(ptf.compounded)
        compounded = set(compounded)

        if len(compounded) == 2:
            raise ValueError(
                "Some portfolios cumulative returns are compounded while some "
                "are non-compounded. You can change the compounded with"
                "`population.set_portfolio_params(compounded=False)`",
            )
        title = "Cumulative Returns"
        compounded = compounded.pop()
        if compounded:
            yaxis_title = f"{title} (rebased at 1000)"
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
            yaxis_title = title
            title = f"{title} (non-compounded)"

        df = pd.concat(cumulative_returns, axis=1).iloc[:, idx]
        df.columns = deduplicate_names(names)

        fig = df.plot()
        fig.update_layout(
            title=title,
            xaxis_title="Observations",
            yaxis_title=yaxis_title,
            legend_title_text="Portfolios",
        )
        if compounded:
            fig.update_yaxes(tickformat=".0f")
        else:
            fig.update_yaxes(tickformat=".2%")
        if log_scale:
            fig.update_yaxes(type="log")
        return fig

    def plot_composition(
        self,
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
        display_sub_ptf_name: bool = True,
    ) -> go.Figure:
        """Plot the compositions of the portfolios in the population.

        Parameters
        ----------
        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        display_sub_ptf_name : bool, default=True
            If this is set to True, each sub-portfolio name composing a multi-period
            portfolio is displayed.

        Returns
        -------
        plot : Figure
            Returns the plotly Figure object.
        """
        df = self.composition(
            names=names, tags=tags, display_sub_ptf_name=display_sub_ptf_name
        ).T
        fig = px.bar(df, x=df.index, y=df.columns)
        fig.update_layout(
            title="Portfolios Composition",
            xaxis={
                "title": "Portfolios",
            },
            yaxis={
                "title": "Weight",
                "tickformat": ",.0%",
            },
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
        names: skt.Names | None = None,
        tags: skt.Tags | None = None,
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

        names : str | list[str], optional
            If provided, the population is filtered by portfolio names.

        tags : str | list[str], optional
            If provided, the population is filtered by portfolio tags.

        Returns
        -------
        plot : Figure
            Returns the plotly Figure object.
        """
        portfolios = self.filter(names=names, tags=tags)
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
            for portfolio in portfolios
        ]
        # Improved formatting
        columns = [str(e) for e in columns]
        hover_data = {str(k): v for k, v in hover_data.items()}

        df = pd.DataFrame(res, columns=columns)
        df["tag"] = df["tag"].astype(str).replace("None", "")

        if show_fronts:
            fronts = self.non_denominated_sort(first_front_only=False)
            if tags is not None:
                ValueError("Cannot plot front with tags selected")
            df["front"] = str(-1)
            for i, front in enumerate(fronts):
                for idx in front:
                    df.iloc[idx, -1] = str(i)
            color = df.columns[-1]
        elif color_scale is not None:
            color = str(color_scale)
        else:
            color = "tag"

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
                            title=str(z),
                            titleside="top",
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
                    symbol="tag",
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
                symbol="tag",
            )
            fig.update_traces(marker_size=10)
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
                legend=dict(yanchor="top", y=0.96, xanchor="left", x=1.25),
            )
        return fig
