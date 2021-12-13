# ANLP Project#18 - Benchmarking of CDCR models

This repository is for my project in the course `Applied Natural Language Processing`

So far, you can start the whole program via `python predict_model.py`.\
You will need to enter the following information:
+ Which dataset, do you want to use - ECB+ or MEANTime
+ Do you want to train/test on a single topic?
+ If yes, which one.

This will load either all data (or only the wanted topic) from a ConLL file into a data structure called `Corpus`.\
Each `Corpus` contains the different `Topics` (or a single topic, if wanted).\
Each `Topic` holds a dictionary of the different `Documents`.\
Each `Document` holds all of its `Sentences`.\
Each `Token` is a representative of the word itself.

But so far each of these is missing the correct annotated mentions.\
In order to add the mentions to the data, the `MentionExtractor.py` needs to be run, but this is still WIP.
<br>
<br>
## Next Steps:
+ Check, if the written mentions after running the `MentionExtractor.py` are correct.
  + The written mentions so far only contain all tokens which should belong to this mention.
+ Add each created mention to the correct sentence of the corpus.
+ Try to test the model of [Barhom et al.](https://github.com/shanybar/event_entity_coref_ecb_plus) (further called EECDCR), with the selfmade dataset corpus.
  + from: "Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution"

The next steps will be updated the next time a new milestone is hit.