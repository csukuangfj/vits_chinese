---
license: apache-2.0
language:
- zh
---

# 下载模型

请去 https://huggingface.co/jackyqs/vits-aishell3-175-chinese/tree/main
下载模型。 

# 3500.txt
3500.txt is downloaded from
https://github.com/kaienfr/Font/blob/master/learnfiles/%E5%B8%B8%E7%94%A8%E6%B1%89%E5%AD%97%E5%BA%93%203500.txt


# aishell3数据介绍:

希尔贝壳中文普通话语音数据库AISHELL-3的语音时长为85小时88035句，可做为多说话人合成系统。录制过程在安静室内环境中， 使用高保真麦克风（44.1kHz，16bit）。

218名来自中国不同口音区域的发言人参与录制。专业语音校对人员进行拼音和韵律标注，并通过严格质量检验，此数据库音字确率在98%以上。

vits模型介绍：

这是一个基于vits_chinese和aishell3 175人中文训练的预训练模型，可以直接用于微调语音克隆，大大缩短微调训练的时间。

该模型使用tesla T4 16G训练了大概2周，500K步，单人语音数据微调1-3小时，即可达到非常逼真的效果，是MOS值最接近真实值的一个模型。

该模型包含了两个模型文件，一个是D_AISHELL.pth，另外一个是G_AISHELL.pth，共同构成了预训练模型。

微调:

需要将这个两个模型文件放到utils.save_checkpoint目录下：

utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))

utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))

推理:

使用通过个人语音数据微调后的G_AISHELL.pth即可。

utils.load_checkpoint("G_pretrained.pth", net_g, None)
