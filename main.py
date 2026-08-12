from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status

from clusters import ClustersUnavailableError, cluster_status
from mapping import load_active_features
from models import Health, Prediction, Pupil
from prediction import PredictionError, predict_pupil
from settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app_settings = settings or Settings.from_env()
    app = FastAPI(title="Career Prediction API", version="1.0.0")

    @app.get("/health", response_model=Health)
    def health() -> Health:
        clusters = cluster_status(
            app_settings,
            load_active_features(app_settings.mapping_path),
        )
        return Health(
            status="ok",
            clusters=clusters.status,
            clusters_updated_at=clusters.updated_at,
        )

    @app.post("/predict", response_model=Prediction)
    def predict(pupil: Pupil) -> Prediction:
        try:
            return predict_pupil(pupil, app_settings)
        except ClustersUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "CLUSTERS_UNAVAILABLE", "message": str(error)},
            ) from error
        except PredictionError as error:
            logging.exception("prediction failed pupilId=%s", pupil.pupil_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "PREDICTION_FAILED", "message": "Prediction could not be calculated"},
            ) from error
        except Exception as error:
            logging.exception("prediction failed pupilId=%s", pupil.pupil_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "PREDICTION_FAILED", "message": "Prediction could not be calculated"},
            ) from error

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
