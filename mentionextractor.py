import os.path

from conll_reader import readConll
from shared.CONSTANTS import dataset_path
from shared.classes import Corpus


def _findSubMention(save_data, data, j):
    momentary_data = data[j]
    mention = momentary_data.split("\t")[4].split("|")
    for ment in mention:
        if ment.startswith("start"):
            mention = ment
            break

    tokens = [save_data[j]]
    data[j] = data[j].replace(f"start{mention}", "SUBBED")
    for k in range(j+1, len(data)):
        # assumption: there is no -| TAG) or TAG | -
        if "start" in data[k] and f"start{mention}" not in data[k]:
            tokens.append(save_data[k])
            _findSubMention(save_data, data, k)
        elif "-" in data[k]:
            tokens.append(save_data[k])
            data[k].replace("-", "SUBBED")
        elif "SUBBED" in data[j]:
            tokens.append(save_data[j])
        elif f"end{mention}" in data[k]:
            tokens.append(save_data[k])
            data[k] = data[k].replace(f"end{mention}", "SUBBED", 1)
            with open("./data/output/output.txt", 'a', encoding="utf8") as f:
                f.write(f"Tokens for annotation: {mention}:\n")
                for token in tokens:
                    f.write(f"{token}\n")
                f.write("\n\n")
            break


def extractMentionsFromConLL(datasetName: str, corpus: Corpus, topicLevel: bool, topicName: str):
    with open(dataset_path[datasetName], "r", encoding="utf8") as f:
        data = f.read()

    data = data.split("\n")

    if topicLevel:
        data = [x for x in data if x.split("\t")[0].startswith(topicName + "/")]

    possible_mentions = []
    for bs in data:
        possible_mention = bs.split("\t")[4]
        if possible_mention != "-":
            # if the annotation looks like this: ACT16236402809085484| ACT16236402809085484
            # or this: -| HUM16236184328979740
            if "|" in possible_mention:
                splitted_mention = possible_mention.split("|")
                for pos_mention in splitted_mention:
                    pos_mention = pos_mention.replace("(", "").replace(")", "").replace(" ", "")
                    if pos_mention not in possible_mentions and pos_mention != "-":
                        possible_mentions.append(pos_mention)
            else:
                # if there is only one annotation
                possible_mention = possible_mention.replace("(", "").replace(")", "")
                if possible_mention not in possible_mentions:
                    possible_mentions.append(possible_mention)
    # To ensure the removal of duplicate entries we cast the list to a set and back to a list
    possible_mentions = list(set(possible_mentions))

    """
        We will overwrite the data over and over again. So in order to create the correct tokens the original
        data needs to be save (in save_data) and everytime we save something, we will save the same index from
        the saved_data.
    """
    save_data = data.copy()
    for mention in possible_mentions:
        """
            Data "pre processing"
        """
        for i, datum in enumerate(data):
            if f"({mention})" in datum:
                with open("./data/output/output.txt", 'a', encoding="utf8") as f:
                    f.write(f"Single data for {mention}: {save_data[i]}\n")
                data[i] = data[i].replace(f"({mention})", "SKIP!")
            data[i] = data[i].replace(f"({mention}", f"start{mention}")
            data[i] = data[i].replace(f"{mention})", f"end{mention}")

        for i, datum in enumerate(data):
            datum_mentions = datum.split("\t")[4]
            datum_mentions = datum_mentions.replace(" ", "").split("|")
            datum_mentions = datum_mentions[::-1]
            for datum_mention in datum_mentions:
                if f"start{mention}" in datum_mention:
                    tokens = [save_data[i]]
                    data[i] = data[i].replace(f"start{mention}", "DONE")
                    for j in range(i+1, len(data)):
                        # assumption: there is no -| TAG) or TAG | -
                        if "start" in data[j] and f"start{mention}" not in data[j]:
                            tokens.append(save_data[j])
                            _findSubMention(save_data, data, j)
                        elif "-" in data[j]:
                            tokens.append(save_data[j])
                            data[j].replace("-", "DONE")
                        elif "SUBBED" in data[j]:
                            tokens.append(save_data[j])
                            data[j].replace("SUBBED", "DONE")
                        elif "DONE" in data[j]:
                            tokens.append(save_data[j])
                        if f"end{mention}" in data[j]:
                            if "DONE" not in data[j]:
                                tokens.append(save_data[j])
                            data[j] = data[j].replace(f"end{mention}", "DONE", 1)
                            with open("./data/output/output.txt", 'a', encoding="utf8") as f:
                                f.write(f"Tokens for annotation: {mention}:\n")
                                for token in tokens:
                                    f.write(f"{token}\n")
                                f.write("\n\n")
                            break

        # entry_split = entry.split("\t")
        # topic_name = entry_split[0].split("/")[0]
        # document_name = entry_split[0].split("/")[1]
        # sent_id = entry_split[1]
        # token_id = entry_split[2]
        # token = entry_split[3]
        # sing_mention = Mention(document_name, sent_id, token_id)
    return corpus


if __name__ == '__main__':
    """
        Since we open a file in append mode, the file would get very large and might not be helpful to check anymore.
        So we remove the file on start up and get a file with only the output of the last test run.
    """
    if os.path.exists("./data/output/output.txt"):
        os.remove("./data/output/output.txt")

    extractMentionsFromConLL("MEANTime", readConll("MEANTime"), True, "corpus_airbus")
