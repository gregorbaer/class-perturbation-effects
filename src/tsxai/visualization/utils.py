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

    This class provides three distinct color palettes optimized for different
    visualization needs:
    - A two-color palette ideal for binary comparisons
    - A five-color palette for small-scale comparisons (2-5 items)
    - A ten-color palette for larger-scale comparisons (6-10 items)

    Each palette is carefully selected to ensure color distinctiveness and
    visual clarity in scientific publications.

    Example:
        Create palettes of different sizes:
        >>> # Binary comparison (2 colors)
        >>> binary_palette = ScientificPalette.get_palette(2)
        >>>
        >>> # Small comparison (4 categories)
        >>> small_palette = ScientificPalette.get_palette(4)
        >>>
        >>> # Large comparison (8 categories)
        >>> large_palette = ScientificPalette.get_palette(8)
    """

    # Two distinct colors, optimized for binary comparisons
    TWO_COLOR_PALETTE = [
        "#006699",  # Nature Blue - Used in Nature journal
        "#DC2830",  # Nature Red - Used in Nature journal
    ]

    # Five distinct colors, optimized for small comparisons
    FIVE_COLOR_PALETTE = [
        "#E69F00",  # Orange - High visibility
        "#56B4E9",  # Light blue - Cool tone
        "#009E73",  # Green - Mid tone
        "#CC79A7",  # Pink/Rose - Warm tone
        "#0072B2",  # Dark blue - Strong accent
    ]

    # Ten distinct colors, optimized for larger comparisons
    TEN_COLOR_PALETTE = [
        "#4C72B0",  # Strong blue - Primary emphasis
        "#55A868",  # Sage green - Natural tone
        "#C44E52",  # Muted red - Warm contrast
        "#8172B3",  # Muted purple - Cool accent
        "#CCB974",  # Light brown - Neutral tone
        "#64B5CD",  # Light blue - Soft accent
        "#666666",  # Dark gray - Neutral contrast
        "#B47846",  # Earth brown - Warm neutral
        "#8C8C8C",  # Medium gray - Mid neutral
        "#6B8E23",  # Olive green - Natural accent
    ]

    @classmethod
    def get_palette(cls, num_colors: int) -> List[str]:
        """Get an appropriate color palette based on the required number of colors.

        Automatically selects the most appropriate palette based on the number of
        colors requested:
        - For num_colors=2: Returns the binary comparison palette
        - For num_colors≤5: Returns colors from the small-scale palette
        - For num_colors≤10: Returns colors from the large-scale palette

        Args:
            num_colors: Number of distinct colors needed (must be between 1 and 10)

        Returns:
            List of hex color codes optimized for the requested number of colors

        Raises:
            ValueError: If num_colors is not between 1 and 10

        Example:
            >>> # Get 4 colors for comparing different categories
            >>> colors = ScientificPalette.get_palette(4)
            >>> print(colors)  # Returns first 4 colors from FIVE_COLOR_PALETTE
            ['#E69F00', '#56B4E9', '#009E73', '#CC79A7']
        """
        if not 1 <= num_colors <= 10:
            raise ValueError("Number of colors must be between 1 and 10")

        if num_colors <= 2:
            return cls.TWO_COLOR_PALETTE
        elif num_colors <= 5:
            return cls.FIVE_COLOR_PALETTE[:num_colors]
        else:
            return cls.TEN_COLOR_PALETTE[:num_colors]
