import cssl

trainer = cssl.Trainer(config_path="config/classifier_cifar_class.yaml")
trainer.pretrain()