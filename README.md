# IDEAL
## IDEA Language model
IDEAL is a language translation model that retokenizes GPT2 tokens into it's own tokenization space. It consists of a bidirectional encoder layer that takes in GPT2 Tokens, and for every token it predicts a token in it's own vocabulary with its own embeddings. It does this with a gumbel softmax. Then, these tokens are passed through an embedding layer, and a standard transformer decoder block follows. 

![image](https://github.com/user-attachments/assets/bad57da5-389b-4b7c-867e-f99ea7f932e4)

## Results

This is a test with a idea vocabulary size of 1024.  
![image](https://github.com/user-attachments/assets/dfcdf1ab-d5fe-4d8c-9280-f4b8f8de467c)


Many more updates to come!
