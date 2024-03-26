from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from io import BytesIO
from langchain_core.documents import Document
from langchain_community.document_loaders.blob_loaders.schema import Blob
class BytesIOPyPDFLoader(PyPDFLoader):
    """Load `PDF` files using `PyMuPDF` from a BytesIO stream."""

    def __init__(
        self,
        pdf_stream: BytesIO,
        *,
        extract_images: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize with a BytesIO stream."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "`PyMuPDF` package not found, please install it with "
                "`pip install pymupdf`"
            )
        # We don't call the super().__init__ here because we don't have a file_path.
        self.pdf_stream = pdf_stream
        self.extract_images = extract_images
        self.text_kwargs = kwargs

    def load(self, **kwargs: Any) -> list[Document]:
        """Load file."""

        text_kwargs = {**self.text_kwargs, **kwargs}

        # Use 'stream' as a placeholder for file_path since we're working with a stream.
        blob = Blob.from_data(self.pdf_stream.getvalue(), path="stream")

        parser = PyPDFParser(extract_images=self.extract_images
        )

        return parser.parse(blob)