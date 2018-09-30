from nltk.stem import WordNetLemmatizer
import logging
import argparse

def main(args):
    fname = "/home/binhnguyen/PycharmProjects/StatisticWork/data/eventMap.txt"
    train_data = "/home/binhnguyen/PycharmProjects/StatisticWork/data/train.txt"

    # 0 = Subtype-event
    TYPE_EVENT = args.event_type

    map_event = dict()
    event_trigger = dict()
    event_lemma = dict()
    single_meaning = dict()
    single_meaning_lm = dict()
    list_trigger = []
    list_lemma = []
    base_events = []
    lemmatizer = WordNetLemmatizer()
    occurence_trigger = dict()
    occurence_lemma_trigger = dict()

    sample_single_meaning = 0
    sample_single_meaning_lm = 0
    trigger_total = 0
    event_other = 0

    with open(fname) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    for sentence in content:
        if len(sentence) != 0:
            list_word = sentence.split(' ')
            base_events.append(list_word[1])
            map_event[list_word[0].replace(list_word[1] + ":", "")] = list_word[1]

    base_events.append("Other")
    map_event["Other"] = "Other"

    content = open(train_data).readlines()
    trigger_total = len(content)

    for sample in content:
        tokens = sample.strip().split("\t")
        event = tokens[1]
        if (TYPE_EVENT):
            event = map_event[event]
        trigger = tokens[3]
        lemma_trigger = lemmatizer.lemmatize(trigger)

        if (event == "Other"):
            event_other += 1
            continue

        if (trigger in occurence_trigger):
            occurence_trigger[trigger] += 1
        else:
            occurence_trigger[trigger] = 1

        if (lemma_trigger in occurence_lemma_trigger):
            occurence_lemma_trigger[lemma_trigger] += 1
        else:
            occurence_lemma_trigger[lemma_trigger] = 1

        if trigger in event_trigger:
            if event != event_trigger[trigger]:
                single_meaning[trigger] = "false"
        else:
            event_trigger[trigger] = event
            single_meaning[trigger] = "true"

        if lemma_trigger in event_lemma:
            if event != event_lemma[lemma_trigger]:
                single_meaning_lm[lemma_trigger] = "false"
        else:
            event_lemma[lemma_trigger] = event
            single_meaning_lm[lemma_trigger] = "true"

    numb_sm_trigger = 0
    numb_sm_lemma_trigger = 0

    for (key, value) in single_meaning.items():
        if (value == "true"):
            numb_sm_trigger += 1
            sample_single_meaning += occurence_trigger[key]

    for (key, value) in single_meaning_lm.items():
        if (value == "true"):
            numb_sm_lemma_trigger += 1
            sample_single_meaning_lm += occurence_lemma_trigger[key]

    level = logging.INFO
    format = '  %(message)s'
    handlers = [logging.FileHandler('statistic_log.txt'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=format,  filename='statistic_log.txt')

    logging.info("---------------------")
    logging.info("Number of subtype label = " + str(len(base_events)))
    if (TYPE_EVENT):
        logging.info("Base Event:")
    else:
        logging.info("Subtype Event:")
    logging.info("Total trigger = " + str(trigger_total))
    logging.info("- Number of single meaning trigger = {}".format(numb_sm_trigger))
    logging.info("- Number of single meaning lemma trigger = {}".format(numb_sm_lemma_trigger))
    logging.info("- Number of \'Other\' event = {}".format(event_other))
    logging.info(
        "- Ratio of sample single meaning trigger = {} %".format(sample_single_meaning * 100.0 / trigger_total))
    logging.info("- Ratio of sample single meaning lemma trigger = {} %".format(
        sample_single_meaning_lm * 100.0 / trigger_total))
    logging.info("- Ratio of other trigger = {}%".format(event_other * 100.0 / trigger_total))
    logging.info("---------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_type', type=int, default=0)
    args = parser.parse_args()
main(args)



