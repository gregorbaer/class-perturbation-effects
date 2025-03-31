from typing import List


def get_plot_dimensions(
    output_type: str = "paper",
    width_type: str = "full",
    aspect_ratio: float = 0.618,
    scale: float = 2.0,
    text_width_mm: float = 122.0,
) -> tuple[float, float]:
    """Get figure dimensions optimized for different output formats.

    Args:
        output_type (str): Target format:
            - "paper": LNCS proceedings (122mm text width)
            - "ppt_wide": PowerPoint 16:9
            - "ppt_std": PowerPoint 4:3
        width_type (str): For papers only:
            - "full": Full text width
            - "wide": 3/4 of text width
            - "narrow": 0.48 of text width
        aspect_ratio (float): Height/width ratio:
            - 0.618: (default) inverse golden ratio
            - 0.5-0.65: wide format
            - 0.75-1.0: square-ish
        scale (float): Development scaling factor.
        text_width_mm (float): Textwidth of the target publication venue.
            Will be used to scale height based on the aspect ratio.
            Defaults to 122mm, which corresponds to LCNS textwidth.

    Returns:
        tuple[float, float]: Width and height in plotting units.
    """
    mm_to_points = 72 / 25.4

    if output_type.startswith("ppt"):
        # PowerPoint optimized sizes (in mm)
        if output_type == "ppt_wide":
            width_mm = 250  # Good width for 16:9 slides
        else:  # ppt_std
            width_mm = 200  # Good width for 4:3 slides

        # For presentations, slightly wider aspect ratio often works better
        height_mm = width_mm * 0.65  # Slightly wider than paper
    else:
        width_presets = {
            "full": text_width_mm,
            "wide": text_width_mm * 0.75,
            "narrow": text_width_mm * 0.48,
        }
        width_mm = width_presets[width_type]
        height_mm = width_mm * aspect_ratio

    return (width_mm * mm_to_points * scale, height_mm * mm_to_points * scale)


class ScientificPalette:
    """Collection of optimized color palettes for scientific visualization.

    Palettes were selected and pre-defined to be color-blind safe.
    Okabe-Ito palette retrieved from: https://jfly.uni-koeln.de/color/#pallet.
    Paul Tol palettes retrieved from: https://sronpersonalpages.nl/~pault.
    """

    # Okabe-Ito Blue, Orange, Red
    OKABE_ITO_PALETTE = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # bluish green
        "#D55E00",  # vermillion
        "#56B4E9",  # sky blue
        "#CC79A7",  # reddish purple
        "#000000",  # black
        "#F0E442",  # yellow
    ]
    PAUL_TOL_BRIGHT = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]
    PAUL_TOL_VIBRANT = [
        "#EE7733",
        "#0077BB",
        "#33BBEE",
        "#EE3377",
        "#CC3311",
        "#009988",
        "#BBBBBB",
    ]
    PAUL_TOL_HIGH_CONTRAST = ["#004488", "#DDAA33", "#BB5566"]

    @classmethod
    def get_palette(
        cls, n_colors: int = None, palette_name: str = "okabe-ito"
    ) -> List[str]:
        """Retrieve color palette with first n_colors.

        Args:
            n_colors: Number of colors to return. If None, returns the full palette.
            palette_name: Name of the palette to use.
                One of 'okabe-ito', 'pt-bright', 'pt-vibrant', 'pt-high-contrast'.
                These palettes were selected

        Returns:
            List of hex color codes from the requested palette.
        """
        if palette_name == "okabe-ito":
            pallete = cls.OKABE_ITO_PALETTE
        elif palette_name == "pt-bright":
            pallete = cls.PAUL_TOL_BRIGHT
        elif palette_name == "pt-vibrant":
            pallete = cls.PAUL_TOL_VIBRANT
        elif palette_name == "pt-high-contrast":
            pallete = cls.PAUL_TOL_HIGH_CONTRAST
        else:
            raise ValueError("Palette name not recognized.")

        assert len(pallete) >= n_colors, "Palette does not have enough colors"

        if n_colors is None:
            return pallete
        else:
            return pallete[:n_colors]


def theme_paper(
    legend_position=None,
    legend_key_height=None,
    legend_key_width=None,
    font_family="Helvetica",
    base_size=8,
    scale=2.0,
):
    """Create a theme for academic paper plots.

    This theme is optimized for academic papers and works well with the
    get_plot_dimensions function, especially when scale=2.0 is used.

    Args:
        legend_position (str): Position of the legend ('none', 'right', 'bottom', 'left', 'top').
            Defaults to None as many paper figures include separate legends.
        legend_key_height (float): Height of legend keys.
        legend_key_width (float): Width of legend keys.
        font_family (str): Font family to use for all text elements.
            Defaults to "Helvetica" which is good for academic papers.
            Other options include "Times New Roman", "Palatino", or "Computer Modern".
        base_size (float): Base font size in points. Defaults to 8.
        scale (float): Scaling factor to match the plot dimensions.
            Should match the scale parameter used in get_plot_dimensions.

    Returns:
        lets_plot.theme: A theme object suitable for academic papers.
    """
    from lets_plot import element_text, theme

    # Adjust text sizes based on scale
    title_size = base_size * 1.4 * scale
    subtitle_size = base_size * 1.2 * scale
    axis_title_size = base_size * 1.1 * scale
    axis_text_size = base_size * scale
    legend_title_size = base_size * 1.1 * scale
    legend_text_size = base_size * scale
    strip_text_size = base_size * 1.1 * scale  # For facet titles

    return theme(
        # Title and subtitle
        plot_title=element_text(family=font_family, size=title_size),
        plot_subtitle=element_text(family=font_family, size=subtitle_size),
        # Axis titles
        axis_title=element_text(family=font_family, size=axis_title_size),
        # Axis text (tick labels)
        axis_text=element_text(family=font_family, size=axis_text_size),
        # Legend
        legend_title=element_text(family=font_family, size=legend_title_size),
        legend_text=element_text(family=font_family, size=legend_text_size),
        # Facet titles
        strip_text=element_text(family=font_family, size=strip_text_size),
        # Additional styling for papers
        legend_position=legend_position,
        legend_key_height=legend_key_height,
        legend_key_width=legend_key_width,
    )


def theme_presentation(
    legend_position=None,
    legend_key_height=None,
    legend_key_width=None,
    font_family="Helvetica",
    base_size=12,
    scale=2.0,
):
    """Create a theme for presentation plots.

    This theme is optimized for slides and presentations and works well
    with the get_plot_dimensions function when used with ppt output types.

    Args:
        legend_position (str): Position of the legend ('none', 'right', 'bottom', 'left', 'top').
            Defaults to None as many presentations include separate legend annotations.
        legend_key_height (float): Height of legend keys.
        legend_key_width (float): Width of legend keys.
        font_family (str): Font family to use for all text elements.
            Defaults to "Arial" which is good for presentations.
        base_size (float): Base font size in points. Defaults to 12.
        scale (float): Scaling factor to match the plot dimensions.
            Should match the scale parameter used in get_plot_dimensions.

    Returns:
        lets_plot.theme: A theme object suitable for presentations.
    """
    from lets_plot import element_text, theme

    # Adjust text sizes based on scale
    title_size = base_size * 1.5 * scale  # Larger title for presentations
    subtitle_size = base_size * 1.3 * scale
    axis_title_size = base_size * 1.2 * scale
    axis_text_size = base_size * scale
    legend_title_size = base_size * 1.2 * scale
    legend_text_size = base_size * scale
    strip_text_size = base_size * 1.2 * scale  # For facet titles

    return theme(
        # Title and subtitle
        plot_title=element_text(family=font_family, size=title_size),
        plot_subtitle=element_text(family=font_family, size=subtitle_size),
        # Axis titles
        axis_title=element_text(family=font_family, size=axis_title_size),
        # Axis text (tick labels)
        axis_text=element_text(family=font_family, size=axis_text_size),
        # Legend
        legend_title=element_text(family=font_family, size=legend_title_size),
        legend_text=element_text(family=font_family, size=legend_text_size),
        # Facet titles
        strip_text=element_text(family=font_family, size=strip_text_size),
        # Additional styling for presentations
        legend_position=legend_position,
        legend_key_height=legend_key_height,
        legend_key_width=legend_key_width,
    )
