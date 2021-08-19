import pandas as pd

description = pd.Series(dtype=str, name="Supplemental Data")
description["N"] = "Number of on-target AOIs"
description["Nc"] = "Number of control off-target AOIs"
description["F"] = "Number of frames"
description["P"] = "Number of pixels along x- and y-axes of an AOI"
description["height"] = "Spot intensity"
description["width"] = "Spot width"
description["background"] = "Image background"
description["offset"] = "Camera offset"
description["pi"] = "Average target-specific binding probability"
description["lamda"] = "Non-specific binding rate"
description["proximity"] = "Proximity parameter"
description[
    "p(specific)"
] = "Average probability of there being any target-specific spot in an AOI image"
description["95% UL"] = "95% CI upper-limit"
description["95% LL"] = "95% CI lower-limit"
description["SNR"] = "Signal-to-noise ratio"
description["MCC"] = "Matthews correlation coefficient"
description["TP"] = "True positives"
description["FN"] = "False negatives"
description["TN"] = "True negatives"
description["FP"] = "False positives"


def resize_columns(writer, df, sheetname):
    worksheet = writer.sheets[sheetname]
    for idx in range(len(df.columns) + 1):  # loop through all columns
        worksheet.set_column(idx, idx, 15)  # set column width
