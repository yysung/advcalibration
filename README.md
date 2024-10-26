### Train buzzer
```
python buzzer.py --guesser_type=gpr --limit=50000 --gpr_guesser_filename=../models/buzztrain_gpr_cache   --questions=../data/qanta.buzztrain.json.gz --buzzer_guessers gpr --logistic_buzzer_filename=models/Length_Frequency_Blank_Capital_Pronouns_Of_Plural_Space_Disam_Diffculty_Year --features Length Frequency Blank Capital Pronouns Of Plural Space Disam Difficulty Year
```
### Evaluate buzzer
```
python eval.py --guesser_type=gpr   --limit=-1   \
--questions=../data/qanta.buzzdev.json.gz --buzzer_guessers gpr \
--gpr_guesser_filename=../models/buzzdev_gpr_cache    \
--logistic_buzzer_filename=models/Length_Frequency_Blank_Capital_Pronouns_Of_Plural_Space_Disam_Diffculty_Year --features Length Frequency Blank Capital Pronouns Of Plural Space Disam Difficulty Year
```
