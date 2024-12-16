# Refrence

https://youtu.be/XlO7iQMV3Ik?si=sGxu-WSAdNqH23oG

```sh
torch-model-archiver --model-name my-model --version 1.0 --model-file model.py --serialized-file my-module.pt --handler handler.py  
```

```sh
torchserve --model-store model-dir/
torchserve --model-store model-dir/
torchserve --model-store model-dir/ --models my-model=my-model.mar --ts-config config.properties
torchserve --model-store model-dir/ --models my-model=img-latex.mar --ts-config config.properties
torchserve --model-store model-dir/ --models my-model=img-latex.mar --ts-config config.properties
```


#--extra-files tacotron.zip,nvidia_tacotron2pyt_fp32_20190306.pth


TypeError: Expected state_dict to be dict-like, got <class 'torch.jit._script.RecursiveScriptModul