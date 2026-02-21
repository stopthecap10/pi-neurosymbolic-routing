#cd ~/pi-neurosymbolic-routing
source .venv/bin/activate
mkdir -p outputs

python3 src/log_run_energy.py \
  --system phi2_q8_grammar \
  --tier T1 \
  --csv_used data/industry_tier1_40.csv \
  --model_file /home/stopthecap10/edge-ai/models/phi-2.Q8_0.gguf \
  --repeats 3 \
  --out_csv outputs/T1_phi2_q8_grammar_r3.csv \
  --trials_csv outputs/T1_phi2_q8_grammar_r3_trials.csv \
  -- \
  python3 src/run_phi2_server_runner_safe.py \
    --csv data/industry_tier1_40.csv \
    --out outputs/T1_phi2_q8_grammar_r3.csv \
    --trials_out outputs/T1_phi2_q8_grammar_r3_trials.csv \
    --server_url http://127.0.0.1:8080/completion \
    --timeout_s 20 \
    --repeats 3 \
    --warmup_per_prompt 1 \
    --n_pred_num 12 \
    --n_pred_log 6 \
    --num_grammar_file grammars/grammar_phi2_answer_int_nolead.gbnf \
    --yesno_grammar_file grammars/grammar_phi2_answer_yesno_strict_final.gbnf