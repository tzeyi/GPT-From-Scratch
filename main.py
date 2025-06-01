import torch
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024, # Max input to model
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False # No Biases in attention Head
}


def generate_text_simple(model, index, max_new_tokens, context_length):
    # index is (batch, n_tokens) array of indices in current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10, then only the last 5 tokens are used as context
        index_cond = index[:, -context_length:]

        # Predict
        with torch.no_grad():
            logits = model(index_cond)

        # Focus only on the last time step (of the token)
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the index of the vocab entry with the highest probability value
        index_next = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        index = torch.cat((index, index_next), dim=1)  # (batch, n_tokens+1)

    return index


def main():
    tokenizer = tiktoken.get_encoding("gpt2") # BPE encoder

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print(f"encoded: {encoded}")

    # Unsqueeze add extra dim at 0, essentially adding a batch dim
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(f"encoded_tensor.shape: {encoded_tensor.shape}")


    model.eval() # evaluation mode, disable dropout for deterministic and stable output

    output = generate_text_simple(
        model=model,
        index=encoded_tensor,
        max_new_tokens=6,
        context_length=GPT_CONFIG_124M["context_length"]
    )

    print("Output:", output)
    # Output length (10) = Original input (4) + Max newly generated token (6)
    print(f"Output length: {len(output[0])}")

    decoded_text = tokenizer.decode(output.squeeze(0).tolist())
    print(decoded_text)

if __name__ == "__main__":
    main()

