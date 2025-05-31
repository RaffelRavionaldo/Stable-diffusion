# Train stable diffusion

The code was on train folder, it was edited version of https://github.com/huggingface/diffusers/tree/main/examples/dreambooth code, The main difference between my code and hugging face's code is that my code can use more than 1 class to train and each image has its own prompt.

I use google colab for train this model because on colab we got free top tier GPU (3-4 hours, Nvidia T4), but if you have it a nice GPU, you can try the code on your PC.

Before go to how to install and fine-tuning the stable diffusion with the code, you need to prepare the data with this structure : 

```
Data/
├─ Person/
├── person1.jpg
├── person1.txt
├── person2.jpg
├── person2.txt
├─ Cat/
├── cat1.jpg
├── cat1.txt
├── cat2.jpg
├── cat2.txt
Prior_data/ (Optional)
├─ Person/
├── Random_person1.jpg
├── Random_person2.jpg
├── classes.txt
├─ Cat/
├── Random_cat1.jpg
├── Random_cat2.jpg
├── classes.txt
```

you can only use 1 class. each image has its prompt (and can be a different object too! like person1 is my picture, so on .txt I write it as "sks_Raffel" or I can make it more complete like "sks_Raffel, smiling, using glasses" and person2 is other person) and on the classes.txt at prior_data, input the class_prompt like "a photo of person" in the Person folders and "a photo of cat" in Cat folders.

~~ a bit out of the subject ~~
on the dataset file, i pyt sks on the .txt. From what I read, this is used to make the model learn new tokens. I suggest you use it if your name is common (like Michael, Ali, etc.).

for use the code, we need install some depedencies, here we are : 

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q accelerate transformers ftfy tensorboard Jinja2 peft ninja --upgrade
pip install xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes
pip install -q git+https://github.com/huggingface/diffusers.git
pip install -q opencv-contrib-python
pip install -q controlnet_aux
```

after install it, you can run this code for starting fine-tuning stable diffusion : 

```
!accelerate launch train_dreambooth_2.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="/content/gdrive/MyDrive/stable diffusion/dataset_2" \
  --output_dir="/content/gdrive/MyDrive/stable diffusion/model_2" \
  --instance_prompt="a photo of [V*]" \
  --class_prompt="a photo of a person" \
  --class_data_dir='/content/gdrive/MyDrive/stable diffusion/prior_data' \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --max_train_steps=4000 \
  --mixed_precision="fp16" \
  --checkpointing_steps=2000 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --with_prior_preservation \
  --train_text_encoder
```

change the path to your exactly where the data is, --class_data_dir and --class_data is optional (you need prior data to run this), experiment with learning_rate, max_train_step and etc.

~~ Another notice ~~
prepare the empty google drive if you want to try a same code with me, because each chekpoint have around 5GB file size, you can not save the checkpoint by make the value on it more than max_train_steps.

