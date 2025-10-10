# Equation of the relationship between lambda_S1 and lambda_S2 based on the cell height
import os

import numpy as np
import matplotlib.pyplot as plt

from src import PROJECT_DIRECTORY

# Define the original wing disc height
original_wing_disc_height = 15.0 # in microns
cell_heights = np.linspace(0, 30, 10000)
cell_heights_to_show = np.array([0.01, 0.1, 0.5, 1.0, 2.0]) * original_wing_disc_height

lambdaS1_normalised = 0.5 + 0.5 * (1 - np.exp(-0.8 * cell_heights ** 0.4))
lambdaS2_normalised = 1 - lambdaS1_normalised

# Lambda values based on the normalised values and the cell height
lambdaS1 = 1.47 * ((cell_heights / original_wing_disc_height) ** 0.25) * lambdaS1_normalised
lambdaS2 = 1.47 * ((cell_heights / original_wing_disc_height) ** 0.25) * lambdaS2_normalised
lambdaS3 = lambdaS1

# Aspect Ratio energy barrier adapted to cell height. IMPORTANT: The initial gradient should be equivalent to the original size.
lambdaR = 8e-7 * (cell_heights / original_wing_disc_height) ** 1.5

# LambdaV adapted to cell height
lambdaV = 1 * (cell_heights / original_wing_disc_height) ** 0.25

plots_to_show = ['lambdaS1', 'lambdaS2', 'lambdaR', 'lambdaV']

for plot_to_show in plots_to_show:
    # Plot the equations as linear values from 0 to max cell height
    plt.figure(figsize=(10, 6))

    # Plot the different lambda values as smooth lines and as a function of cell height without markers
    if plot_to_show == 'lambdaS1':
        plt.plot(cell_heights, lambdaS1, label=r'$\lambda_{S1}=\lambda_{S3}$', color='blue')
        plt.ylabel(r'$\lambda_{S1}=\lambda_{S3}$', fontsize=20)
    elif plot_to_show == 'lambdaS2':
        plt.plot(cell_heights, lambdaS2, label=r'$\lambda_{S2}$', color='orange')
        plt.ylabel(r'$\lambda_{S2}$', fontsize=20)
    elif plot_to_show == 'lambdaR':
        plt.plot(cell_heights, lambdaR, label=r'$\lambda_{R}$', color='red')
        plt.ylabel(r'$\lambda_{R}$', fontsize=20)
    elif plot_to_show == 'lambdaV':
        plt.plot(cell_heights, lambdaV, label=r'$\lambda_{V}$', color='purple')
        plt.ylabel(r'$\lambda_{V}$', fontsize=20)

    # Create vertical dashed lines for the cell heights to show and add text with the parameter value
    for ch in cell_heights_to_show:
        plt.axvline(x=ch, color='gray', linestyle='--', linewidth=0.8)
        # Get the value of the parameter at that cell height
        if plot_to_show == 'lambdaS1':
            value = 1.47 * ((ch / original_wing_disc_height) ** 0.25) * (0.5 + 0.5 * (1 - np.exp(-0.8 * ch ** 0.4)))
        elif plot_to_show == 'lambdaS2':
            value = 1.47 * ((ch / original_wing_disc_height) ** 0.25) * (1 - (0.5 + 0.5 * (1 - np.exp(-0.8 * ch ** 0.4))))
        elif plot_to_show == 'lambdaR':
            value = 8e-7 * (ch / original_wing_disc_height) ** 1.5
        elif plot_to_show == 'lambdaV':
            value = 1 * (ch / original_wing_disc_height) ** 0.25

        plt.text(ch, value, f'{value:.1e}', fontsize=12, color='black', ha='right', va='bottom', rotation=90)

    # Add legend and labels. Log scale for y axis
    plt.xlabel('Aspect Ratio (AR)', fontsize=20)
    plt.xlim(0, 30)
    plt.ylim(0,)
    # Add to the xticks some minor ticks at the cell heights to show
    plt.xticks(rotation=45, fontsize=20, fontweight='bold', ticks=cell_heights_to_show)
    plt.yticks(fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIRECTORY, 'Result', 'parameters_based_on_cell_height_' + plot_to_show + '.png'))
    plt.close()

