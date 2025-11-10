#!/bin/bash

# Step 1: Go to submodule
cd external/lstm-breakout-predictor

# Step 2: Add and commit changes in submodule
git add simulator/output/simulated_trades.xlsx
git commit -m "Auto-Update: simulated_trades.xlsx with feedback"
git push origin master

# Step 3: Go back to parent repository
cd ../../..

# Step 4: Update submodule reference in parent repository
git add external/lstm-breakout-predictor
git commit -m "Updated submodule lstm-breakout-predictor after XLS update"
git push origin main
echo "Submodule updated and changes pushed to parent repository."
