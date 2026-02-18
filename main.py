from train import main as train_main
from evaluate import main as evaluate_main

def main():
    # Train models
    train_main()
    
    # Evaluate and visualize results
    evaluate_main()

if  __name__  == "__main__":
    main()