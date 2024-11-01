import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 config: dict,  # تأكد من وجود فاصلة هنا
                 tb_logger=None  # tb_logger معامل اختياري
                 ) -> None:
        super(VAEXperiment, self).__init__()

        self.config = config
        self.tb_logger = tb_logger  # حفظ tb_logger
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = params.get('retain_first_backpass', False)
        self.automatic_optimization = False  # تعطيل التحسين التلقائي
        """
        ll_values: refers to Reconstruction Loss values. These are the losses saved from reconstructing the input data.
        kl_values: refers to KLD Loss (Kullback-Leibler Divergence Loss). These are the losses saved from the difference
        between two distributions.
        """
        self.ll_values = []  # لقيم Reconstruction Loss
        self.kl_values = []  # لقيم KLD Loss



#class VAEXperiment(pl.LightningModule):

#    def __init__(self,
#                 vae_model: BaseVAE,
#                 params: dict
#                 config, tb_logger=None
#                 ) -> None:
#        super(VAEXperiment, self).__init__()

#        self.config = config
#        self.tb_logger = tb_logger ###################ADD
#        self.model = vae_model
#        self.params = params
#        self.curr_device = None
#        #self.hold_graph = False
#        self.hold_graph = params.get('retain_first_backpass', False)
#        self.automatic_optimization = False  # Disable automatic optimization
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        #return self.model(input, **kwargs)
        return self.model(input, tb_logger=self.tb_logger, **kwargs)


    #def training_step(self, batch, batch_idx, optimizer_idx = 0):
    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              #optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        ###########################
        # حفظ القيم في قوائم للرجوع إليها لاحقًا
        self.ll_values.append(train_loss['Reconstruction_Loss'].item()) 
        self.kl_values.append(train_loss['KLD'].item()) 
        ########################### 


        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
    ################################
    
        # Manually perform optimization
        opt = self.optimizers()  # إذا كان لديك محسن واحد
        opt.zero_grad()
        self.manual_backward(train_loss['loss'])
        opt.step()
        #return train_loss['loss']
        # Return loss and latest ll and kl values
        return train_loss['loss']
        #, self.ll_values[-1], self.kl_values[-1]
######################################
        # إضافة دوال الاسترجاع
    def get_ll_values(self):
        return self.ll_values

    def get_kl_values(self):
        return self.kl_values
    ###############################################

        #opt1, opt2 = self.optimizers()
        #opt1.zero_grad()
        #opt2.zero_grad()
        #self.manual_backward(train_loss['loss'])
        #opt1.step()
        #opt2.step()
    #######################################

        

    #def validation_step(self, batch, batch_idx, optimizer_idx = 0):
    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            #optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        #recons = self.model.generate(test_input, labels = test_label)
        recons = self.model.generate(test_input, tb_logger=self.tb_logger, labels=test_label)############## ADD tb_logger

        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
