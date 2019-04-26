# pytorch_trainer
a general trainer for class/segmentation task

### Usage
<pre>
model_trainer = trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                    loss_criterion=criterion, eval_criterion=IoU, device=device,
                    dataloaders=dataloader, max_epochs=200, verbose_train = 1, verbose_val=100,
                    ckpt_frequency=1000, checkpoint_dir='checkpoints', max_iter=10000000,
                    comments='training a model with hyper-params: xxxxxx')
model_trainer.train()
<pre>
