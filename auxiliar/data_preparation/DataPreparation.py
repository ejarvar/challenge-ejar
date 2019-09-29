import json
import pandas as pd


class DataPreparation:
    def __init__(self, config_file):
        """
        Data preparation class
        """
        self.config = self.read_json(config_file)

    def read_json(self, file_name):
        with open(file_name, "r") as f:
            data = json.load(f)
        return data

    def read_scraping(self, file_name):
        data = self.read_json(file_name)
        blogs = data.get("blog")
        for b in blogs:
            b.update({"url": b.get("url").split("/")[-2]})
        return pd.DataFrame(blogs)

    def __f_url__(self, x):
        for sc in self.scraping_filter:
            if sc in x.split("/"):
                return sc
        return False

    def __f_title__(self, x):
        if "not found" in x.lower():
            return False
        else:
            return x

    def __f_time__(self, x):
        if x == "<00:00:01":
            return 0
        hour, minute, second = x.split(":")
        seconds = int(hour) * 3600 + int(minute) * 60 + int(second)
        return seconds

    def read_analytics(self, file_name):
        data = pd.read_csv(file_name, sep=",")
        preprocess_url = data["Page"].apply(self.__f_url__)
        preprocess_title = data["Page Title"].apply(self.__f_title__)
        preprocess_time = data["Avg. Time on Page"].apply(self.__f_time__)
        # TODO: we need the length of the blog text

        data.update(preprocess_url)
        data.update(preprocess_title)
        data.update(preprocess_time)

        filter_url = data["Page"] != False
        filter_title = data["Page Title"] != False

        data = data[filter_url][filter_title]

        return data

    def preprocess_scraping(self):
        path_scraping = self.config.get("path_scraping")
        if path_scraping is None:
            raise Exception("Scraping file not found")

        scraping = self.read_scraping(path_scraping)
        self.scraping_filter = scraping["url"].tolist()

        return scraping

    def preprocess_analytics(self):
        path_analytics = self.config.get("path_analytics")
        if path_analytics is None:
            raise Exception("Analytics file not found")

        analytics = self.read_analytics(path_analytics)

        return analytics


if __name__ == "__main__":
    bbva = DataPreparation("/mnt/c/GitEjar/Personal/BBVAChallenge/config.json")
    path_processed = bbva.config.get("path_processed")
    bbva.preprocess_scraping().to_csv(
        "{}/processed_scraping.csv".format(path_processed), sep="|"
    )
    bbva.preprocess_analytics().to_csv(
        "{}/processed_analytics.csv".format(path_processed), sep="|"
    )
