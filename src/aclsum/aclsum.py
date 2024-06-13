from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import sienna

DATA_DIR = Path(__file__).parent / "dataset"
ASPECTS = ["challenge", "approach", "outcome"]
SECTIONS = ["abstract", "introduction", "conclusion"]


@dataclass
class Document:
    """A data point in ACLSum dataset"""

    id: str  # Document ID corresponding to ACL Anthology
    title: str  # Title of the paper
    summaries: dict[str, str]  # Summaries on three aspects (aspect -> summary)
    sentences: dict[
        str, list[str]
    ]  # Sentences segmented by each section (section -> sentences)
    highlights: dict[
        str, dict[str, list[int]]
    ]  # Binary labels for highlighted sentences, 1 is highlighted (aspect -> section -> sentence indices)

    def get_all_sentences(self, sections: list[str] | None = None) -> list[str]:
        """Get all sentences of the paper. By passing sections names, you can extract sentences from specific sections.

        Parameters
        ----------
        sections : list[str] | None
            Specific sections to extract sentences from. When `None`, get sentences from all the sections.

        Returns
        -------
        list[str]
        """
        sections = SECTIONS if sections is None else sections
        return [sent for section in sections for sent in self.sentences[section]]

    def get_all_highlights(
        self, aspect: str, sections: list[str] | None = None
    ) -> list[int]:
        """Get all the binary labels for highlight annotation. By passing sections names, you can extract sentences from specific sections.

        Parameters
        ----------
        aspect : str
            Specify the aspect (challenge, approach, outcome) to extract the labels for.
        sections : list[str] | None
            Specific sections to extract sentences from. When `None`, get sentences from all the sections.

        Returns
        -------
        list[int]
        """
        sections = SECTIONS if sections is None else sections
        return [
            label for section in sections for label in self.highlights[aspect][section]
        ]

    def get_all_highlighted_sentences(
        self, aspect: str, sections: list[str] | None = None
    ) -> list[str]:
        """Get all the highlighted sentences. Different from the `get_all_highlights` function, this
        returns sentences not labels.

        Parameters
        ----------
        aspect : str
            Specify the aspect (challenge, approach, outcome) to extract the labels for.
        sections : list[str] | None
            Specific sections to extract sentences from. When `None`, get sentences from all the sections.

        Returns
        -------
        list[str]
        """
        sections = SECTIONS if sections is None else sections
        sents = self.get_all_sentences(sections)
        labels = self.get_all_highlights(aspect, sections)
        assert len(sents) == len(labels)
        return [sent for sent, label in zip(sents, labels) if label == 1]

    def get_fulltext_parse(self) -> dict[str, Any]:
        return sienna.load(str(DATA_DIR / "paper_jsons" / f"{self.id}.json"))


@dataclass
class ACLSum:
    """The main class to work with ACLSum dataset.

    Attributes
    ----------
    documents : list[Document]
        List of each data sample.
    split : Literal["train", "val", "test"]
    """

    documents: list[Document]
    split: Literal["train", "val", "test"]

    def __init__(self, split: Literal["train", "val", "test"]):
        d = sienna.load(str(DATA_DIR / f"{split}.jsonl"))
        documents: list[Document] = []
        for x in d:
            documents.append(
                Document(
                    id=x["id"],
                    title=x["title"],
                    summaries=x["summary"],
                    sentences=x["sentences"],
                    highlights=x["highlights"],
                )
            )
        self.documents = documents
        self.split = split

    def __getitem__(self, idx: int) -> "Document":
        return self.documents[idx]

    def __iter__(self):
        for doc in self.documents:
            yield doc

    def __len__(self) -> int:
        return len(self.documents)

    def to_parallel_corpus(
        self, aspect: Literal["challenge", "approach", "outcome"], do_extractive: bool
    ) -> tuple[list[str], list[str]]:
        srcs, tgts = [], []
        for document in self.documents:
            srcs.append(" ".join(document.get_all_sentences()))
            tgts.append(
                document.summaries[aspect]
                if not do_extractive
                else " ".join(document.get_all_highlighted_sentences(aspect))
            )
        return srcs, tgts
