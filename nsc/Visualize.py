# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 23:04:22 2022

@author: jkcle
"""
import matplotlib
import numpy as np
import colorsys

from matplotlib import colors as mc

from .GeneralizedMean import kappa_to_r
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.manifold import TSNE
from tensorflow import reshape
import tensorflow_probability as tfp
tfd = tfp.distributions
Normal = tfd.Normal


class Visualize:

    def __init__(self,
                 z_dim,
                 kappa,
                 input_type,
                 sample_func,                 
                 z_sample,
                 sample,
                 sample_labels, 
                 gmean_metrics,
                 gmean_log_prob_values, 
                 display_path='.', 
                 parameter_str=''
                 ):
        self.z_dim = z_dim
        self.kappa = kappa
        self.input_type = input_type
        self.sample_func = sample_func        
        self.z_sample = z_sample
        self.sample_labels = sample_labels
        self.gmean_metrics = gmean_metrics
        self.gmean_log_prob_values = gmean_log_prob_values
        self.display_path = display_path
        self.parameter_str = parameter_str
        if len(z_sample) <= 16:
            sample_images = sample_func(z_sample)  # predictions
            sample_images_orig = sample
        else:
            sample_images = sample_func(z_sample)[:16]  # first 16 predictions
            print(sample_images.shape)
            sample_images_orig = sample[:16]

        self.sample_images = sample_images
        self.sample_images_orig = sample_images_orig
        self.digit_size = len(sample_images[0])
        return
    
    def display(self, show=True, **kwargs):
        self.display_generated_images(show=show, **kwargs)
        self.display_original_images(show=show, **kwargs)
        self.display_latent_space(show=show, **kwargs)
        #self.display_manifold(show=show, **kwargs)
        self.display_histogram(key='recon', show=show, **kwargs)
        self.display_histogram(key='kldiv', show=show, **kwargs)
        self.display_histogram(key='elbo', show=show, **kwargs)
        return


    def display_test(self, show=True, **kwargs):
        self.display_generated_images(show=show, **kwargs)
        self.display_original_images(show=show, **kwargs)
        self.display_latent_space(show=show, **kwargs)
        #self.display_manifold(show=show, **kwargs)
        
        self.display_histogram(key='recon', show=show, **kwargs)
        self.display_histogram(key='kldiv', show=show, **kwargs)
        self.display_histogram(key='elbo', show=show, **kwargs)
        
        return
    
    def display_generated_images(self, image_size=4, show=True, **kwargs):

        fig = plt.figure(figsize=(image_size, image_size))
        for i in range(self.sample_images.shape[0]):
            plt.subplot(image_size, image_size, i + 1)
            plt.imshow(self.sample_images[i,:,:,:])
            plt.axis('off')
            # tight_layout minimizes the overlap between 2 sub-plots
            plt.axis('Off')
            plt.savefig(f"{self.display_path}/generated_images/" + \
                       f"{self._picture_name('images', **kwargs)}"
                    )
        if show:
            plt.show();
        return


    def display_original_images(self, image_size=4, show=True, **kwargs):

        fig = plt.figure(figsize=(image_size, image_size))
        for i in range(self.sample_images_orig.shape[0]):
            plt.subplot(image_size, image_size, i + 1)
            plt.imshow(self.sample_images_orig[i, :, :, :])
            plt.axis('off')
            # tight_layout minimizes the overlap between 2 sub-plots
            plt.axis('Off')
            plt.savefig(f"{self.display_path}/generated_images/" + \
                       f"{self._picture_name('original_images', **kwargs)}"
                    )
        if show:
            plt.show();
        return
    
    def display_latent_space(self, image_size=4, show=True, **kwargs):
        # display a 2D plot of the digit classes in the latent space
        
        colors = ['pink', 'red', 'orange', 'yellow', 'green', 
        'blue', 'purple', 'brown', 'gray', 'black']
        plt.figure(figsize=(image_size, image_size))
        z_dim = self.z_dim
        if z_dim==2:
            plt.scatter(
                self.z_sample[:, 0], 
                self.z_sample[:, 1], 
                c=self.sample_labels,
                cmap=matplotlib.colors.ListedColormap(colors)
                )
            plt.xlabel('z[0]')
            plt.ylabel('z[1]')
        
        else:
            encoded_imgs_embedded = TSNE(n_components=2).fit_transform(
                self.z_sample
                )
            plt.scatter(
                encoded_imgs_embedded[:,0],
                encoded_imgs_embedded[:,1],
                c=self.sample_labels,
                cmap=matplotlib.colors.ListedColormap(colors)
                )
            plt.xlabel("t-SNE 1st dimension")
            plt.ylabel("t-SNE 2nd dimension")
              
            plt.colorbar()
            plt.savefig(f"{self.display_path}/latent_spaces/" + \
                        f"{self._picture_name('latent', **kwargs)}"
                        )
        if show:
            plt.show();
        return
    
    def display_manifold(self, n=20, image_size=10, show=True, **kwargs):
        """Plots n x n digit images decoded from the latent space."""
        norm = Normal(0, 1)
        grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
        grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
        image_width = self.digit_size*n
        image_height = image_width
        image = np.zeros((image_height, image_width))
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = np.array([[xi, yi]])
                x_decoded = self.sample_func(z)
                digit = reshape(
                    x_decoded[0], 
                    (self.digit_size, self.digit_size)
                    )
                image[i*self.digit_size: (i+1)*self.digit_size,
                      j*self.digit_size: (j+1)*self.digit_size] = digit.numpy()
        plt.figure(figsize=(image_size, image_size))
        plt.imshow(image, cmap='Greys_r')
        plt.axis('Off')
        plt.savefig(f"{self.display_path}/manifolds/" + \
        f"{self._picture_name('mani', **kwargs)}"
        )
        if show:
            plt.show();
        return

    def display_histogram(self, key, image_size=(24, 10), show=True, **kwargs):
        """
        Plots histogram of log-probabilities, log_prob_values. 
        Label x-axis with log10 scale so that the xtick location resides at 
        prob_value = exp(log_prob_values). Then exp(log_prob_values) = 10^power,
        or log_prob_values = power*log(10)  
        """
        # Set histogram title, labels, and ticks: edited August 5 BT (in process)
        if key == 'recon':
            xlabel = 'Reconstruction Likelihood in Logscale'
            power0 = -1000
            xtick_powers_base_10=np.array(range(power0, 0, 100))
            xtick_labels = [r'$10^{'+str(k)+'}$' for k in xtick_powers_base_10]
            xticks =  xtick_powers_base_10*np.log(10)
            xlim_values = [power0*np.log(10), 0]
        elif key == 'kldiv':
            xlabel = 'KL Divergence Likelihood in Logscale'
            power0 = -14
            power1 = -11
            xtick_powers_base_10=np.array(np.arange(power0, power1, .5))
            xtick_labels = [r'$10^{'+str(k)+'}$' for k in xtick_powers_base_10]
            xticks =  xtick_powers_base_10*np.log(10)
            xlim_values = [power0*np.log(10), power1*np.log(10)]
        else:
            xlabel = 'ELBO Likelihood in Logscale'
            power0 = -1000
            xtick_powers_base_10=np.array(range(power0, 0, 100))
            xtick_labels = [r'$10^{'+str(k)+'}$' for k in xtick_powers_base_10]
            xticks =  xtick_powers_base_10*np.log(10)
            xlim_values = [power0*np.log(10), 0]

        # Retrieve the generalized means and metric values
        decisiveness = self.gmean_metrics[f'{key}_decisiveness']
        accuracy = self.gmean_metrics[f'{key}_accuracy']
        robustness = self.gmean_metrics[f'{key}_robustness']
        r = kappa_to_r(float(self.kappa), dim=1)  # TODO USE Z-DIM?
        r = round(r, 3)
        r_gen_mean = self.gmean_metrics[f'{key}_{r}_generalized_mean']
        log_prob_values = self.gmean_log_prob_values[key]

        # Exporting to csv
        DataFrame(log_prob_values).to_csv(
            f"{self.display_path}/histograms/" + \
            f"{self._picture_name(f'hist_{key}', **kwargs)}.csv"
            )

        # The histogram of the data
        fig, ax = plt.subplots(figsize=(12, 5))
        xtick_axis = [np.log(xtick) for xtick in xticks]
        #xtick_labels = [str(xtick) for xtick in xticks]
        #ax.set_xticks(xtick_axis)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        
        # Plot in logscale, so convert the metric as logs as well
        log_dec = np.log(decisiveness)
        log_acc = np.log(accuracy)
        log_rob = np.log(robustness)
        log_r_gen_mean = np.log(r_gen_mean)
        dec_txt = f'{decisiveness:0.2e}'
        acc_txt = f'{accuracy:0.2e}'
        rob_txt = f'{robustness:0.2e}'
        gen_mean_txt = f'{r_gen_mean:0.2e}'
        plt.axvline(log_dec, color='r', linestyle='dashed', linewidth=2)
        plt.text(log_dec, 10*10, dec_txt, color='r', size='large', weight='bold')
        plt.axvline(log_acc, color='b', linestyle='dashed', linewidth=2)
        plt.text(log_acc, 10*30, acc_txt, color='b', size='large', weight='bold')
        plt.axvline(log_rob, color='g', linestyle='dashed', linewidth=2)
        plt.text(log_rob, 10*60, rob_txt, color='g', size='large', weight='bold')

        if float(self.kappa) != 0:
          plt.axvline(log_r_gen_mean, color='k', linestyle='dashed', linewidth=2)
          plt.text(log_r_gen_mean, 10*100, gen_mean_txt, color='k', size='large', weight='bold')
        
        plt.hist(
            log_prob_values, 
            log=True,
            bins=100, 
            facecolor='white', 
            edgecolor='black'
            )

        plt.xlabel(xlabel, fontdict = {'fontsize' : 20, 'weight': 'normal'})
        #plt.xlim(xlim_values)
        ax.set_xlim(xmin=xlim_values[0], xmax=xlim_values[1])

        plt.ylabel(
            "Frequency in logscale", 
            fontdict = {'fontsize' : 20, 'weight': 'normal'}
            )
         
        plt.savefig(
            f"{self.display_path}/histograms/" + \
            f"{self._picture_name(f'hist_{key}', **kwargs)}"
            )
        
        if show:
            plt.show();

        return
    
    #Changed 8_26_2021 "+ parameter_str"
    def _picture_name(self, display_type, **kwargs):
        name = display_type + self.parameter_str
        for key, value in kwargs.items():
            name += f'_{key}{str(value)}'
        return f'{name}.png'
    

    ##### plotting functions
    def plot_latent_images(model, n, digit_size=28):
        """Plots n x n digit images decoded from the latent space."""
        norm = tfp.distributions.Normal(0, 1)
        grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
        grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
        image_width = digit_size*n
        image_height = image_width
        image = np.zeros((image_height, image_width))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = np.array([[xi, yi]])
                x_decoded = model.sample(z)
                digit = reshape(x_decoded[0], (digit_size, digit_size))
                image[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit.numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='Greys_r')
        plt.axis('Off')
        plt.show();
        return

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_training(vae_dict, metric='neg_elbo'):
  for key, vae in vae_dict.items():
      metric_df = pd.concat([vae.metrics_df, vae.val_metrics_df], axis=1)
      x = metric_df['epoch']
      y = metric_df[f'train_{metric}']
      plt.plot(x, y, label=f'Training {metric}')
      y = metric_df[f'val_{metric}']
      plt.plot(x, y, label=f'Validation {metric}')
      plt.xticks(x)
      plt.legend()
      plt.xlabel('Epoch')
      plt.ylabel(f'{metric}')
      plt.title(f'Training and Validation {metric} vs. Epochs')
      plt.show()
  return

