import re
import pandas as pd
import itertools


class LikwidParser:

    # NOTE: Achitecture dependent and probably should be adjusted depending on it
    groups = {
        "Metric": {
            "L3CACHE": ["L3 request rate", "L3 miss rate", "L3 miss ratio"],
            "MEM": [
                "Memory data volume",
                "Memory read bandwidth",
                "Memory write bandwidth",
                "Memory bandwidth",
            ],
            "CACHES": [
                "L3 to/from system bandwidth",
                "L3 to/from system data volume",
                "System to L3 bandwidth",
                "System to L3 data volume",
                "L3 to system bandwidth",
                "L3 to system data volume",
                "L1 to/from L2 bandwidth",
                "L1 to/from L2 data volume",
                "L3 to L2 load bandwidth",
                "L3 to L2 load data volume",
                "L2 to L3 evict bandwidth",
                "L2 to L3 evict data volume",
                "L2 to/from L3 bandwidth",
                "L2 to/from L3 data volume",
                "Memory write data volume",
                "Memory read data volume",
            ],
        }
    }

    header = "p  q      cells        dofs  timeCGit throughput itCG    timeMV"

    def __init__(self, filename):

        f = open(filename)
        content = f.read()
        likwid_sections = re.findall("Region ((.*?)_(\d+)),", content)
        sections = list({_[1] for _ in likwid_sections})
        sizes = list({int(_[2]) for _ in likwid_sections})
        sizes.sort()

        self.filename = filename
        self.sections = sections
        self.sizes = sizes
        self.flat_groups = list(
            itertools.chain(*(v for k, v in LikwidParser.groups.get("Metric").items()))
        )
        self.likwid_data = {
            s: {
                "STAT": pd.DataFrame(columns=self.flat_groups, index=self.sizes),
                "SUM": pd.DataFrame(columns=self.flat_groups, index=self.sizes),
            }
            for s in sections
        }
        self.benchmark_data = pd.DataFrame()
        self.parse()

    def get_number_from_csv(self, extract_type, numbers):

        # need to summarize across different sockets
        if extract_type == "SUM":
            number = sum(0 if x == "nil" else float(x) for x in numbers)
        else:
            tmp = re.compile(r"(.*?),")
            matches = tmp.findall("Metric,Sum,Min,Max,Avg,")
            matched_index = matches.index("Avg")
            index_to_extract = matched_index
            number = float(numbers[index_to_extract])
        return number

    def parse(self):

        with open(self.filename, "r") as f:
            all_lines = f.readlines()
            for i, line in enumerate(all_lines):

                # ---Parse benchmark header (not likwid specific) --
                if self.header in line:
                    regwsv = re.compile(r"(.*?)\s+")
                    header_items = [i for i in regwsv.findall(line) if i]
                    dofs_id = header_items.index("dofs")
                    while True and i < len(all_lines) - 1:
                        i = i + 1
                        line = all_lines[i]
                        items = [i for i in regwsv.findall(line) if i]
                        if len(items) < 1 or 'TABLE' in line:
                            break
                        dofs = items[dofs_id]
                        # process standard header field and additional, benchmark
                        # specific
                        for index, h in enumerate(itertools.chain(header_items,
                                                    range(len(items) - len(header_items)))):
                            self.benchmark_data.at[dofs, h] = float(items[index])

                # --- Try to parse actual likwid data --
                # try to get current group
                likwid_group_reg = re.search(r"Metric.*?,(\w+)", line)
                if likwid_group_reg is not None:
                    likwid_group = likwid_group_reg.group(1)
                    if likwid_group in LikwidParser.groups["Metric"]:
                        # find find the size of current region
                        match = re.search("Region ((.*?)_(\d+)),", line)
                        if match:
                            section = match.group(2)
                            size = int(match.group(3))
                            # finally find all the metrics we might need here
                            # and try to get it
                            needed_metrics = LikwidParser.groups["Metric"].get(
                                likwid_group
                            )

                            while True and i < len(all_lines) - 1:
                                i = i + 1
                                line = all_lines[i]
                                regcsv = re.compile(r"(.*?),")
                                items = regcsv.findall(line)
                                if len(items) < 1:
                                    break
                                first, *rest = [i for i in items if i]
                                partial_match = next(
                                    (_ for _ in needed_metrics if _ in first), None
                                )
                                if first == "TABLE":
                                    break
                                if first == "Metric" or not partial_match:
                                    continue

                                is_stat = "STAT" in line
                                extract_type = "STAT" if is_stat else "SUM"

                                number = self.get_number_from_csv(extract_type,rest)
                                self.likwid_data[section][extract_type].at[
                                    size, partial_match
                                ] = number
