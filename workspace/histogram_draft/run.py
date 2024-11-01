import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from Histograms import GeneralizedMean  
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return

    #tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  #name=config['model_params']['name'],)
    
    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    # إضافة Generalized Mean هنا
    # التأكد من وجود القيم قبل استخدامهما
    if 'll_values' in locals() and 'kl_values' in locals():
        g_mean = GeneralizedMean(ll_values, kl_values, kappa, z_dim)


    #model = vae_models[config['model_params']['name']](**config['model_params'])
    #experiment = VAEXperiment(model,
    #                      config['exp_params'])

    ###################################### ADD
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'],
                           config, tb_logger=tb_logger)
    ######################################

#data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    num_devices = 1
    data = VAEDataset(
        **config["data_params"],
        pin_memory=num_devices > 0  
    )

    data.setup()

    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True
            ),
        ],
        accelerator='gpu',  # Specify that I want to use a GPU
        devices=1,           # Number of GPUs to use
        max_epochs=1,
    )
    #runner = Trainer(logger=tb_logger,
#                    callbacks=[
#                        LearningRateMonitor(),
#                        ModelCheckpoint(save_top_k=2, 
#                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
#                                       monitor= "val_loss",
#                                       save_last= True),
#                   ],
#                    #strategy=DDPPlugin(find_unused_parameters=False), #use distributed training across multiple GPUs,
#                   **config['trainer_params'])


    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/z").mkdir(exist_ok=True, parents=True)  # إنشاء مجلد لـ z_samples


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

    #############################

    #############################
   

    # Save value after treaning
    ll_values = experiment.get_ll_values()  # الحصول على القيم المحفوظة لـ Reconstruction Loss
    kl_values = experiment.get_kl_values()  # الحصول على القيم المحفوظة لـ KLD Loss
    

    
    kappa = 0.025  # قم بتعديل القيمة حسب الحاجة
    z_dim = config['model_params']['z_dim']  

    # إنشاء g_mean بعد الحصول على ll_values و kl_values
    g_mean = GeneralizedMean(ll_values, kl_values, kappa, z_dim)

    # استدعاء دالة عرض الهيستوغرام
    g_mean.display_histogram('recon')  # يمكنك استخدام 'kldiv' أو 'elbo' حسب الحاجة
    #g_mean.display_histogram('kldiv')
    #g_mean.display_histogram('elbo')



# كتابة القيم إلى ملف
    with open("loss_values.txt", "w") as file:
        for ll, kl in zip(ll_values, kl_values):
            file.write(f"Reconstruction Loss: {ll}, KLD Loss: {kl}\n")
    ###############################
if __name__ == '__main__':
    main()