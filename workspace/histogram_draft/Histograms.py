import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#converts the value of kappa into the value of r.
def kappa_to_r(kappa, dim=1):
    return -2 * kappa / (1 + dim * kappa)
"""
The replace_tensor_zeros function is used to handle tensor data in PyTorch, specifically by replacing any zero values 
in a tensor with a very small number (10 raised to the power of -300). Here’s a simple breakdown:

Key Points:
Purpose: The function replaces zero elements in a 1D tensor to avoid issues with calculations that could 
happen with zero values, such as errors in division or logarithmic calculations.

Input:

tnsr: This is the input tensor, expected to be a 1D tensor.
Output: The function returns a new tensor where all zero values have been replaced with 
10^-300
 , allowing safe calculations without errors caused by zeros.

Specificity to the Code:
Use Case: Although this function could be used in various situations that involve tensor data, 
it’s mainly designed to prepare tensors for model calculations in this specific code. It ensures 
that the tensors going into further calculations don’t have zeros that could disrupt results.
"""

def replace_tensor_zeros(tnsr, power=-300):
    """Replaces 0 elements in a 1D Tensor with 10^power."""
    idx = torch.where(tnsr == 0)[0]  # الحصول على إندكسات الأصفار
    values = torch.pow(torch.ones(idx.shape, dtype=tnsr.dtype) * 10, power)

    # تحقق إذا كان هناك فهارس صفر
    if idx.numel() == 0:
        return tnsr  # إذا لم يكن هناك فهارس، إرجاع التنسور الأصلي
    
    # إنشاء sparse_tensor
    sparse_tensor = torch.sparse.FloatTensor(
        idx.unsqueeze(0), 
        values.unsqueeze(0), 
        torch.Size((1, tnsr.shape[0]))  # هنا، نحدد الأبعاد بشكل صحيح
    )
    
    new_tnsr = tnsr + sparse_tensor.to_dense()
    return new_tnsr

# This function calculates a generalized mean for a set of values, 
# using the parameter r to adjust how the mean is calculated.
def generalized_mean(values, r):
    inv_n = 1 / values.numel()  # Equivalent to tf.size(values).numpy()
    new_values = replace_tensor_zeros(values)

    #r: A number that controls the calculation method. Changing r changes the result in different ways.
    # How it works:
    # 1. The function first replaces any zeros in `values` with very small numbers (using `replace_tensor_zeros` discussed earlier).
    # 2. Then, it calculates the generalized mean:
    #   If r is not zero, it calculates the mean by summing the values raised to the power of r.
    #   If r is zero, it uses logarithms for the calculation.
    if r != 0:
        gen_mean = torch.pow(inv_n * torch.sum(torch.pow(new_values, r)), 1 / r)
    else:
        gen_mean = torch.exp(inv_n * torch.sum(torch.log(new_values)))
    
    return gen_mean

# This class calculates several types of generalized means for different sets of values, 
# organized to handle metrics like "ELBO," "RECONSTRUCTION," and "DIVERGENCE."
class GeneralizedMean:
    def __init__(self, ll_values, kl_values, kappa, z_dim, display_path="output", parameter_str=''):
        self.kappa = kappa
        self.z_dim = z_dim
        self.display_path = "D:/celeba_data/New folder/PyTorch-VAE-master_histogram"  
        self.gmean_metrics = pd.Series()
        self.gmean_log_prob_values = {}
        self.parameter_str = parameter_str 

        # Main parameters:
        # ll_values and kl_values: The sets of values for which means are calculated.
        #Converts `ll_values` and `kl_values` into Tensors to make calculations easier.
        ll_values = torch.tensor(ll_values, dtype=torch.float32)
        kl_values = torch.tensor(kl_values, dtype=torch.float32)

        """
        Then it computes generalized means for three types:
        ELBO GENERALIZED MEANS: Calculates the mean of combining `ll_values` and `kl_values`.
        RECONSTRUCTION GENERALIZED MEANS: Calculates the mean for `ll_values` only.
        DIVERGENCE GENERALIZED MEANS: Calculates the mean for `kl_values`, using a negative sign.
        """
        print('\nELBO GENERALIZED MEANS')
        self._save_generalized_mean_metrics('elbo', ll_values - kl_values)
        print('\nRECONSTRUCTION GENERALIZED MEANS')
        self._save_generalized_mean_metrics('recon', ll_values)
        print('\nDIVERGENCE GENERALIZED MEANS')
        self._save_generalized_mean_metrics('kldiv', -kl_values)

    """
    This function calculates and stores several metrics for the generalized mean of a specific set of log-probability values, 
    such as "decisiveness," "accuracy," "robustness," and a generalized mean. 
    It also validates these values to ensure they can be safely used in calculations.
    """
    #prob_values: Converts log_prob_values to probabilities by taking the exponential.
    def _save_generalized_mean_metrics(self, key, log_prob_values):
        prob_values = torch.exp(log_prob_values)
        #   If prob_values is empty, a warning is printed, and inv_n is set to 0. 
        #   Otherwise, inv_n is calculated as 1/n, where n is the number of values in prob_values.
        if len(prob_values) == 0:
            print("Warning: prob_values is empty. Cannot calculate inv_n.")
            self.inv_n = 0  # أو قم بإعطاء قيمة افتراضية مناسبة، حسب السياق
        else:
            self.inv_n = 1 / len(prob_values)  # 1/n
        ##   Calculates metrics like decisiveness, accuracy, and robustness
        decisiveness = self._calculate_decisiveness(prob_values)
        accuracy = self._calculate_accuracy(prob_values)
        robustness = self._calculate_robustness(prob_values)
        # The r-value is determined using kappa_to_r, then the generalized mean is computed using the generalized_mean function.
        r = kappa_to_r(float(self.kappa), dim=1)  # TODO USE Z-DIM?
        gen_mean = generalized_mean(prob_values, r)
        """
        a new Pandas Series is created containing the four values: decisiveness, accuracy, robustness, 
        and gen_mean. Each value is assigned an index name using the provided key, allowing the user 
        to identify the meaning of each value
        """
        curr_metrics = pd.Series(
            [decisiveness, accuracy, robustness, gen_mean],
            index=[f'{key}_decisiveness', f'{key}_accuracy', f'{key}_robustness',
                   f'{key}_{round(r, 3)}_generalized_mean']
        )
        """
        The new series curr_metrics is merged with the existing series self.gmean_metrics using pd.concat. 
        This allows all calculated metrics to be gathered in one place.
        Values from log_prob_values are stored with the key, making them easy to access later.
        """
        # استخدام pd.concat بدلاً من append
        self.gmean_metrics = pd.concat([self.gmean_metrics, curr_metrics])
        self.gmean_log_prob_values[key] = log_prob_values
        
        print("NaN in log_prob_values:", torch.isnan(log_prob_values).any())
        print("inf in log_prob_values:", torch.isinf(log_prob_values).any())

        """
        A check is performed for any invalid (NaN) or infinite (inf) values in log_prob_values, and the results are printed.
        """
        if not torch.isfinite(log_prob_values).all():
            print("log_prob_values contains non-finite values.")
        
        log_prob_values = log_prob_values[log_prob_values > 0]  # This line keeps only values greater than 0, ignoring any negative or zero values.
        """
        A list of metrics is created, checked for any infinite values, and if found, those values are set to 0.
        """
        metrics = [decisiveness, accuracy, robustness, gen_mean]
        for metric in metrics:
            if not torch.isfinite(metric):
                print(f"{metric} contains non-finite values.")
                metric = torch.tensor(0)  

        print("log_prob_values:", log_prob_values)
        print("decisiveness:", decisiveness)
        print("accuracy:", accuracy)
        print("robustness:", robustness)
        print("r_gen_mean:", gen_mean)

    #Decisiveness is the arithmetic mean
    def _calculate_decisiveness(self, values):
        result = generalized_mean(values, r=1.0)
        return result
    #Accuracy is the Geometric Mean
    def _calculate_accuracy(self, values):
        result = generalized_mean(values, r=0.0)
        return result
    
    def _calculate_robustness(self, values):
        result = generalized_mean(values, r=-2/3)
        return result
    
    def get_metrics(self):
        return self.gmean_metrics
    
    def get_log_prob_values(self):
        return self.gmean_log_prob_values
    
    def get_next_filename(self, base_dir, base_name):
        """Generates a unique filename by appending a number to the base name if a file already exists."""
        i = 1
        while os.path.exists(f"{base_dir}/{base_name}_{i}.csv"):
            i += 1
        return f"{base_dir}/{base_name}_{i}.csv"
    """
    Function definition and docstring: This function takes a key and optional parameters to display a histogram
    of log-probabilities related to a particular metric. image_size sets the figure size, and show determines 
    if the plot is displayed immediately.
    """
    """
    Setting x-axis labels and ticks based on key: Sets up the xlabel, xtick_labels, and xticks for different types
    of metrics (e.g., 'recon', 'kldiv'). Each section defines a different range for xticks depending 
    on the expected values for each metric type.
    """
    def display_histogram(self, key, image_size=(24, 10), show=True, **kwargs):
        """Plots histogram of log-probabilities."""
        if key == 'recon':
            xlabel = 'Reconstruction Likelihood in Logscale'
            #power0 = -1000
            power0 = -5
            xtick_powers_base_10 = np.array(range(power0, 1, 1))  # تعديلات تناسب القيم الأصغر
            #xtick_powers_base_10 = np.array(range(power0, 0, 100))
            xtick_labels = [f'$10^{{{k}}}$' for k in xtick_powers_base_10]
            xticks = xtick_powers_base_10 * np.log(10)
            xlim_values = [power0 * np.log(10), 0]
        elif key == 'kldiv':
            xlabel = 'KL Divergence'#Likelihood in Logscale'
            power0 = -14
            power1 = -11
            #xtick_powers_base_10 = np.arange(power0, power1, .5)
            xtick_powers_base_10 = np.arange(-5, 1, 0.5)
            xtick_labels = [f'$10^{{{k}}}$' for k in xtick_powers_base_10]
            xticks = xtick_powers_base_10 * np.log(10)
            xlim_values = [power0 * np.log(10), power1 * np.log(10)]

        else:
            xlabel = 'ELBO Likelihood in Logscale'
            power0 = -5
            xtick_powers_base_10 = np.array(range(power0, 0, 100))
            xtick_labels = [f'$10^{{{k}}}$' for k in xtick_powers_base_10]
            xticks = xtick_powers_base_10 * np.log(10)
            #xlim_values = [power0 * np.log(10), np.max(np.log(log_prob_values.cpu().numpy()))]
            xlim_values = [power0 * np.log(10), 0]

        """
        Extracts metric values such as decisiveness, accuracy, robustness, and 
        a generalized mean from self.gmean_metrics, all associated with the specified key.
        """

        # Retrieve the generalized means and metric values
        decisiveness = self.gmean_metrics[f'{key}_decisiveness']
        accuracy = self.gmean_metrics[f'{key}_accuracy']
        robustness = self.gmean_metrics[f'{key}_robustness']
        r = round(kappa_to_r(float(self.kappa), dim=1), 3)
        r_gen_mean = self.gmean_metrics[f'{key}_{r}_generalized_mean']
        log_prob_values = self.gmean_log_prob_values[key]

        """
        Ensures log_prob_values contains data; otherwise, raises an error to notify 
        that no data is available for plotting.
        """
        if log_prob_values.numel() == 0:
            raise ValueError("log_prob_values is empty. Please check the data.")

        """
        Ensures that the histograms directory exists within the display path to store generated files.
        """
        hist_dir = os.path.join(self.display_path, 'histograms')
        os.makedirs(hist_dir, exist_ok=True)

        # Exporting to csv
        csv_filename = self.get_next_filename(hist_dir, self._picture_name(f'hist_{key}', **kwargs))
        pd.DataFrame(log_prob_values.cpu().numpy()).to_csv(csv_filename)

        """
        Specifies the x-axis limits (xlim_values) and y-axis limit (ylim) to control the range of displayed values.
        """
        #xlim_values = [(-10) * np.log(10), 10]
        #xlim_values = [np.log(10), np.max(np.log(log_prob_values.cpu().numpy()))]  # Update xlim based on log_prob_values
        #plt.ylim(0, 1000)

        # Plot histogram
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

        """
        Converts each metric (decisiveness, accuracy, robustness, and generalized mean)
        to log scale for display on the plot.
        """
        log_dec = np.log(decisiveness)
        log_acc = np.log(accuracy)
        log_rob = np.log(robustness)
        log_r_gen_mean = np.log(r_gen_mean)
        dec_txt = f'{decisiveness:0.2e}'
        acc_txt = f'{accuracy:0.2e}'
        rob_txt = f'{robustness:0.2e}'
        gen_mean_txt = f'{r_gen_mean:0.2e}'
        print("log_dec:", log_dec)
        print("log_acc:", log_acc)
        print("log_rob:", log_rob)
        print("log_r_gen_mean:", log_r_gen_mean)

        plt.axvline(log_dec, color='r', linestyle='dashed', linewidth=2)
        plt.text(log_dec, 10*10, dec_txt, color='r', size='large', weight='bold')
        plt.axvline(log_acc, color='b', linestyle='dashed', linewidth=2)
        plt.text(log_acc, 10*30, acc_txt, color='b', size='large', weight='bold')
        plt.axvline(log_rob, color='g', linestyle='dashed', linewidth=2)
        plt.text(log_rob, 10*50, rob_txt, color='g', size='large', weight='bold')
        plt.axvline(log_r_gen_mean, color='k', linestyle='dashed', linewidth=2)
        plt.text(log_r_gen_mean, 10*70, gen_mean_txt, color='k', size='large', weight='bold')

        """
        Ensures y_values (log-probability values) is non-empty after any transformations; otherwise, it raises an error.
        """
        y_values = log_prob_values.cpu().numpy()
        if len(y_values) == 0:
            raise ValueError("y_values is empty after filtering. Please check the log_prob_values.")
        
        

        ax.hist(y_values, bins=200, alpha=0.5)
        plt.xlim([np.min(y_values), np.max(y_values)])
        #plt.xlim([np.log(10), np.max(np.log(log_prob_values[log_prob_values.isfinite()].cpu().numpy()))])
        #plt.xlim(xlim_values)
        plt.ylim(0, 1500)
        plt.xlabel(xlabel)
        plt.ylabel('Counts')
        #plt.title(f'{key} Histograms: Log Probabilities')
        plt.grid()
        plt.tight_layout()
        
        if show:
            plt.show()
        plt.savefig(os.path.join(hist_dir, f"{self._picture_name(f'hist_{key}', **kwargs)}.png"))
        plt.close(fig)

    """
    Defines a method to generate filenames for images, incorporating any additional parameters.
    """

    def _picture_name(self, name, **kwargs):
        """Generates the picture name with additional parameters."""
        return f"{name}_{self.parameter_str}"
