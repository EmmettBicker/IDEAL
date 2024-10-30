# IDEAL
## IDEA Language model

IDEAL is a translation model that converts GPT2 tokens into its own token vocabulary. It uses:

A bidirectional encoder that processes GPT2 tokens
A Gumbel softmax layer that maps each input token to IDEAL's custom vocabulary
An embedding layer followed by a transformer decoder that processes these mapped tokens

IDEAL is a language translation model that converts GPT2 tokens into its own token vocabulary. It uses a bidirectional encoder that takes GPT2 Tokens as input. Embeddings processed by this encoder are then passed through a gumbel softmax to map each embedding into a token in IDEAL's custom ideal vocabulary. These idea tokens are then passed through an embedding layer followed by a transformer decoder that processes these idea embeddings.

![image](https://github.com/user-attachments/assets/f744ab9d-8899-48fc-b1c9-60ab1ea32b84)


## Results

This is a test with an idea vocabulary size of 1024.  
![image](https://github.com/user-attachments/assets/dfcdf1ab-d5fe-4d8c-9280-f4b8f8de467c)


Many more updates to come!
