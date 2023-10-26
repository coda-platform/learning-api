import express from 'express';
import PrepareRequestBody from '../models/request/prepareRequestBody';
import learningServices from '../services/learningServices';
import TrainRequestBody from '../models/request/trainRequestBody';
import EvaluateRequestBody from '../models/request/evaluateRequestBody';

var router = express.Router();

router.post('/prepare', async (req, res, next) => {
    try {
        const body: PrepareRequestBody = req.body;
        const response = await learningServices.getPrepare(body);
        console.log(`⚡️[coda-learning-api]: Launching job with options: \n` +
            response.options ? JSON.stringify(response.options, null, 2) : "Options were empty")
        res.send(response);
    }
    catch (error: any) {
        console.error(error.stack)
        next(error);
    }
});

router.post('/train', async (req, res, next) => {
    try {
        const body: TrainRequestBody = req.body;
        const response = await learningServices.getTrain(body);
        res.send(response);
    }
    catch (error: any) {
        console.error(error.stack)
        next(error);
    }
});

router.post('/evaluate', async (req, res, next) => {
    try {
        const body: EvaluateRequestBody = req.body;
        const response = await learningServices.getEvaluate(body);
        res.send(response);
    }
    catch (error: any) {
        console.error(error.stack)
        next(error);
    }
});

export default router;