import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from utils import classification_metrics, deduplicate_headlines, validate_splits, init_model, clean_headline, compute_metrics

class ReviewFixTests(unittest.TestCase):
    def test_normalized_duplicates_and_conflicts(self):
        df = pd.DataFrame({'title': ['Hello  WORLD', ' hello world ', 'Conflict', 'CONFLICT', 'Other'], 'labels': [0, 0, 0, 1, 1]})
        cleaned = deduplicate_headlines(df)
        self.assertEqual(cleaned['title'].tolist(), ['Hello WORLD', 'Other'])

    def test_cross_split_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits({'train': pd.DataFrame({'title': ['HELLO'], 'labels': [0]}),
                             'test': pd.DataFrame({'title': [' hello '], 'labels': [0]})})

    def test_macro_metrics_with_asymmetric_errors(self):
        metrics = classification_metrics([0, 0, 0, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(metrics['precision_macro'], 0.75)
        self.assertAlmostEqual(metrics['recall_macro'], 5 / 6)
        self.assertAlmostEqual(metrics['f1_macro'], 11 / 15)

    def test_optional_log_loss(self):
        labels = [0, 1]
        predictions = [0, 1]
        self.assertNotIn('log_loss', classification_metrics(labels, predictions))
        metrics = classification_metrics(labels, predictions, [[0.8, 0.2], [0.1, 0.9]])
        self.assertAlmostEqual(metrics['log_loss'], -(np.log(0.8) + np.log(0.9)) / 2)
        single_class = classification_metrics([0, 0], [0, 0], [[0.8, 0.2], [0.9, 0.1]])
        self.assertAlmostEqual(single_class['log_loss'], metrics['log_loss'])

    def test_transformer_log_loss_uses_stable_softmax(self):
        logits = np.array([[1001., 1000.], [-1000., -999.]])
        metrics = compute_metrics((logits, np.array([0, 1])))
        self.assertAlmostEqual(metrics['log_loss'], np.log1p(np.exp(-1)))
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertNotIn('loss', metrics)

    def test_video_marker_variations(self):
        for marker in ['(VIDEO)', '[video]', '( Video )', '[ VIDEOS ]', '{Video}', '（ＶＩＤＥＯ）']:
            with self.subTest(marker=marker):
                self.assertEqual(clean_headline(f"Breaking {marker} news"), "Breaking news")
        self.assertEqual(clean_headline("A video about video games"), "A video about video games")
        self.assertEqual(clean_headline("[VIDEO] (video)"), "")

    def test_tweet_and_live_video_markers(self):
        for marker in ['(TWEETS)', '[tweet]', '{ Tweets }', '(LIVE VIDEO)',
                       '[ live   videos ]', '{Live-Video}', '(live–video)',
                       '[LIVEVIDEO]', '（ＬＩＶＥ ＶＩＤＥＯ）']:
            with self.subTest(marker=marker):
                self.assertEqual(clean_headline(f"Breaking {marker} news"), "Breaking news")
        sentence = "Live video captures tweets about the event"
        self.assertEqual(clean_headline(sentence), sentence)
        self.assertEqual(clean_headline("(tweets about policy)"), "(tweets about policy)")
        self.assertEqual(clean_headline("[TWEETS] (LIVE VIDEO)"), "")

    def test_audio_markers(self):
        for marker in ['(AUDIO)', '[audio]', '{ Audio }', '( audio )', '（ＡＵＤＩＯ）']:
            with self.subTest(marker=marker):
                self.assertEqual(clean_headline(f"Breaking {marker} news"), "Breaking news")
        self.assertEqual(clean_headline("Audio recording released"), "Audio recording released")
        self.assertEqual(clean_headline("(audio recording)"), "(audio recording)")

    def test_image_markers(self):
        for marker in ['(IMAGES)', '(IMAGE)', '[images]', '{ Image }', '( images )', '（ＩＭＡＧＥＳ）']:
            with self.subTest(marker=marker):
                self.assertEqual(clean_headline(f"Breaking {marker} news"), "Breaking news")
        self.assertEqual(clean_headline("Images show the event"), "Images show the event")
        self.assertEqual(clean_headline("(images of the event)"), "(images of the event)")

    def test_details_markers(self):
        for marker in ['(DETAILS)', '(DETAIL)', '[details]', '{ Details }', '( details )', '（ＤＥＴＡＩＬＳ）']:
            with self.subTest(marker=marker):
                self.assertEqual(clean_headline(f"Breaking {marker} news"), "Breaking news")
        self.assertEqual(clean_headline("Details emerge about the event"), "Details emerge about the event")
        self.assertEqual(clean_headline("(details of the event)"), "(details of the event)")

    def test_marker_removal_precedes_deduplication(self):
        df = pd.DataFrame({'title': ['Story (VIDEO)', 'Story', '[VIDEO]'], 'labels': [0, 0, 1]})
        self.assertEqual(deduplicate_headlines(df)['title'].tolist(), ['Story'])

    def test_source_identifiers(self):
        for source in ['Reuters', 'REUTERS', 'The New York Times', 'the new  york times',
                       'InfoWars', 'Info Wars', 'infowars.com', 'Associated Press', 'BBC News']:
            with self.subTest(source=source):
                self.assertEqual(clean_headline(f"[{source}] Story (VIDEO)"), "Story")
                self.assertEqual(clean_headline(f"Story | {source}"), "Story")
        self.assertEqual(clean_headline("AP reports time for video games"),
                         "AP reports time for video games")
        self.assertEqual(clean_headline("CNNish Reutersville"), "CNNish Reutersville")
        self.assertEqual(clean_headline("Reuters reports news"), "reports news")
        cleaned = clean_headline("[Reuters] Story | InfoWars")
        self.assertEqual(clean_headline(cleaned), cleaned)

    def test_composite_media_markers(self):
        for marker in ['(+VIDEO)', '(+ Video)', '(SCREENSHOTS)', '[WATCH]',
                       '(VIDEO/TWEETS)', '(IMAGES/VIDEO)', '(IMAGE/VIDEO)',
                       '[ tweets + screenshots ]', '{VIDEO & IMAGES}', '(video, tweets)']:
            with self.subTest(marker=marker):
                self.assertEqual(clean_headline(f"Story {marker}"), "Story")
        self.assertEqual(clean_headline("(video about policy)"), "(video about policy)")
        self.assertEqual(clean_headline("(VIDEO/UNRELATED)"), "(VIDEO/UNRELATED)")
        self.assertEqual(clean_headline("Watch screenshots appear"), "Watch screenshots appear")

    def test_publisher_attribution_suffixes(self):
        for suffix in ['– TruthFeed', '| New Eastern Outlook', '| EndingFed News Network',
                       '| Daily Wire', '- daily wire', '— TRUTHFEED']:
            with self.subTest(suffix=suffix):
                self.assertEqual(clean_headline(f"Story {suffix}"), "Story")
        self.assertEqual(clean_headline("Story | Daily Wire (VIDEO)"), "Story")
        self.assertEqual(clean_headline("Story | Daily Wire – TruthFeed"), "Story")
        for text in ['Daily Wire reports news', 'Story – here is why',
                     'Story | unknown publisher', 'Story | Daily Wire reports']:
            self.assertEqual(clean_headline(text), text)

    def test_url_only_headlines_are_excluded(self):
        for url in ['https://example.com/news', 'HTTP://example.com/a.jpg',
                    'www.example.com', 'https://a.com https://b.com']:
            self.assertEqual(clean_headline(url), "")
        self.assertEqual(clean_headline("Story https://example.com"), "Story https://example.com")
        df = pd.DataFrame({'title': ['https://example.com', 'Story'], 'labels': [1, 0]})
        self.assertEqual(deduplicate_headlines(df)['title'].tolist(), ['Story'])

    def test_label_mapping_is_inverse(self):
        with patch('utils.AutoModelForSequenceClassification.from_pretrained') as load:
            init_model('unused')
        kwargs = load.call_args.kwargs
        self.assertEqual({v: k for k, v in kwargs['id2label'].items()}, kwargs['label2id'])

if __name__ == '__main__':
    unittest.main()
