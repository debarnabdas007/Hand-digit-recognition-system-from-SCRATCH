from fastapi import HTTPException, status

class ModelLoadError(HTTPException):
    def __init__(self, detail: str = "Machine Learning model failed to load."):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=detail
        )

class InvalidImageError(HTTPException):
    def __init__(self, detail: str = "Invalid image data provided. Must be a Base64 string."):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=detail
        )