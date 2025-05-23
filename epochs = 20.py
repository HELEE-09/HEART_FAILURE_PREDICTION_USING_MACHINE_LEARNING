epochs = 20
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Move input and labels to GPU if available
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_batch)  # Model prediction

        # Compute the loss
        loss = criterion(outputs, y_batch)

        # Backward pass and optimize
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the model parameters

        # Accumulate the loss
        running_loss += loss.item()

    # Print loss for each epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
