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
lambdaS1 = 0.38 * cell_heights ** 0.5 * lambdaS1_normalised
lambdaS2 = 0.38 * cell_heights ** 0.5 * lambdaS2_normalised
lambdaS3 = lambdaS1

# Aspect Ratio energy barrier adapted to cell height. IMPORTANT: The initial gradient should be equivalent to the original size.
p_e_AR = 1.5
lambdaR = 8e-7 * (cell_heights / original_wing_disc_height) ** p_e_AR

# LambdaV adapted to cell height
p_volume = 0.5
lambdaV = 1 * (cell_heights / original_wing_disc_height) ** p_volume

# Purse string strength as a function based on cell height from 0 to 1.
lateral_cables_strength =  (1 - np.exp(-0.8 * (cell_heights - 1) ** 0.4))
purse_string_strength = 1 - lateral_cables_strength

plots_to_show = ['lambdaS1', 'lambdaS2', 'lambdaR', 'lambdaV', 'purse_string_strength']

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
    elif plot_to_show == 'purse_string_strength':
        plt.plot(cell_heights, purse_string_strength, label=r'Purse string ($k_{lt_{ps}})$', color='brown')
        plt.plot(cell_heights, lateral_cables_strength, label=r'Lateral cables ($k_{lt_{lc}})$', color='black')
        plt.ylabel(r'$k_{lt}$', fontsize=20)
        plt.ylim(0, 1)

    # Create vertical dashed lines for the cell heights to show and add text with the parameter value
    for ch in cell_heights_to_show:
        plt.axvline(x=ch, color='gray', linestyle='--', linewidth=0.8)
        # Get the value of the parameter at that cell height
        if plot_to_show == 'lambdaS1':
            value = 0.38 * ch ** 0.5 * (0.5 + 0.5 * (1 - np.exp(-0.8 * ch ** 0.4)))
            if ch < 1:
                plt.text(ch+0.15, value + 0.01, f'{value:.1e}', fontsize=12, color='black', ha='left', va='top', rotation=90)
            elif ch > 15:
                plt.text(ch, value - 0.01, f'{value:.1e}', fontsize=12, color='black', ha='right', va='top', rotation=90)
            else:
                # Italics for scientific notation
                plt.text(ch, value + 0.01, f'{value:.1e}', fontsize=12, color='black', ha='right', va='bottom', rotation=90)
        elif plot_to_show == 'lambdaS2':
            value = 0.38 * ch ** 0.5 * (1 - (0.5 + 0.5 * (1 - np.exp(-0.8 * ch ** 0.4))))
            if ch < 1:
                plt.text(ch+0.1, value, f'{value:.1e}', fontsize=12, color='black', ha='left', va='top', rotation=90)
            else:
                plt.text(ch, value, f'{value:.1e}', fontsize=12, color='black', ha='right', va='top', rotation=90)
        elif plot_to_show == 'lambdaR':
            value = 8e-7 * (ch / original_wing_disc_height) ** p_e_AR
            if ch < 1:
                plt.text(ch+0.1, value + 0.1*1e-6, f'{value:.1e}', fontsize=12, color='black', ha='left', va='bottom', rotation=90)
            elif ch > 15:
                plt.text(ch, value - 0.1*1e-6, f'{value:.1e}', fontsize=12, color='black', ha='right', va='top', rotation=90)
            else:
                plt.text(ch, value + 0.1*1e-6, f'{value:.1e}', fontsize=12, color='black', ha='right', va='bottom', rotation=90)
        elif plot_to_show == 'lambdaV':
            value = 1 * (ch / original_wing_disc_height) ** p_volume
            if ch < 1:
                plt.text(ch+0.1, value, f'{value:.1e}', fontsize=12, color='black', ha='left', va='top', rotation=90)
            elif ch > 15:
                plt.text(ch, value - 0.01, f'{value:.1e}', fontsize=12, color='black', ha='right', va='top', rotation=90)
            else:
                plt.text(ch, value + 0.01, f'{value:.1e}', fontsize=12, color='black', ha='right', va='bottom', rotation=90)
        elif plot_to_show == 'purse_string_strength':
            plt.legend(fontsize=15)

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

