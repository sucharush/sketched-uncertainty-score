#!/bin/bash
trap "echo 'Interrupted'; exit 1" SIGINT 

MODELS=("mlp" "lenet")
SOLVERS=("sl" "ra" "ll" "hl")
OODS=("fashion" "mnist" "kmnist" "rotate")
RUNS=3

for model in "${MODELS[@]}"; do
  # Set steps 
  if [[ "$model" == "mlp" ]]; then
    STEPS=50
  elif [[ "$model" == "lenet" ]]; then
    STEPS=100
  fi

  for solver in "${SOLVERS[@]}"; do
    for ood in "${OODS[@]}"; do
      # Skip ID == OOD
      if [[ "$model" == "mlp" && "$ood" == "mnist" ]]; then
        continue
      fi
      if [[ "$model" == "lenet" && "$ood" == "fashion" ]]; then
        continue
      fi

      echo ">> Running: model=$model, solver=$solver, ood=$ood, steps=$STEPS"
      python -m main.run --model "$model" --solver "$solver" --steps "$STEPS" --ood "$ood" --runs "$RUNS"
      echo ""
    done
  done
done
