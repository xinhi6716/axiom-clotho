"""Two exact raw-height trends with unchanged derived geometry summaries."""

from fractions import Fraction as F

SUM_HEIGHT_DEPTH = 200
REPORTS = ((F(7, 20), F(150), F(65)), (F(1, 5), F(100), F(85)))


def geometry(report, height):
    dilution, hatch, layer = report
    root = {72: F(4, 5), 128: F(3, 5)}[height]
    width = hatch / root
    clad_area = layer * hatch
    melted_area = clad_area * dilution / (1 - dilution)
    return height, SUM_HEIGHT_DEPTH - height, width, clad_area, melted_area


def summarize_squared(row):
    height, depth, width, clad, melted = row
    hatch_squared = width**2 * depth / (height + depth)
    return melted / (clad + melted), hatch_squared, clad**2 / hatch_squared


if __name__ == "__main__":
    for heights in ((72, 128), (128, 72)):
        pair = tuple(geometry(report, h) for report, h in zip(REPORTS, heights))
        print(f"height change={heights[1] - heights[0]:+d}; "
              f"derived summaries reconcile={tuple(map(summarize_squared, pair)) == tuple((d, h*h, t*t) for d, h, t in REPORTS)}")
