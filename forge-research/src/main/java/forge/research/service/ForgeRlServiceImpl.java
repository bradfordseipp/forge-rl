package forge.research.service;

import forge.research.DecisionContext;
import forge.research.RlGameManager;
import forge.research.proto.ForgeRlServiceGrpc;
import forge.research.proto.ResetRequest;
import forge.research.proto.ResetResponse;
import forge.research.proto.SimulateRequest;
import forge.research.proto.SimulateResponse;
import forge.research.proto.StepRequest;
import forge.research.proto.StepResponse;
import io.grpc.stub.StreamObserver;

/**
 * gRPC service implementation for ForgeRlService.
 */
public class ForgeRlServiceImpl extends ForgeRlServiceGrpc.ForgeRlServiceImplBase {

    private final RlGameManager gameManager = new RlGameManager();

    @Override
    public void reset(ResetRequest request, StreamObserver<ResetResponse> responseObserver) {
        try {
            DecisionContext ctx = gameManager.resetGame(
                    request.getDeckPathA(),
                    request.getDeckPathB(),
                    request.getAgentPlayerIndex(),
                    request.getDualRl(),
                    request.getOpponentModelPath(),
                    request.getSeed());

            ResetResponse.Builder response = ResetResponse.newBuilder();
            if (ctx != null) {
                response.setObservation(ctx.getObservation());
                if (ctx.getDecisionPoint() != null) {
                    response.setDecisionPoint(ctx.getDecisionPoint());
                }
            }

            responseObserver.onNext(response.build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            responseObserver.onError(
                    io.grpc.Status.INTERNAL
                            .withDescription("Reset failed: " + e.getMessage())
                            .withCause(e)
                            .asRuntimeException());
        }
    }

    @Override
    public void step(StepRequest request, StreamObserver<StepResponse> responseObserver) {
        try {
            DecisionContext ctx = gameManager.step(request.getActionIndex());

            StepResponse.Builder response = StepResponse.newBuilder();
            if (ctx != null) {
                response.setObservation(ctx.getObservation());
                response.setReward(ctx.getReward());
                response.setTerminated(ctx.isGameOver());
                if (ctx.getDecisionPoint() != null) {
                    response.setDecisionPoint(ctx.getDecisionPoint());
                }
                if (ctx.isGameOver()) {
                    response.setGameResult(gameManager.buildGameResult());
                }
            }

            responseObserver.onNext(response.build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            responseObserver.onError(
                    io.grpc.Status.INTERNAL
                            .withDescription("Step failed: " + e.getMessage())
                            .withCause(e)
                            .asRuntimeException());
        }
    }

    @Override
    public void simulate(SimulateRequest request, StreamObserver<SimulateResponse> responseObserver) {
        try {
            RlGameManager.SimulationResult result = gameManager.simulate(
                    request.getFirstActionIndex(),
                    request.getMaxAgentDecisions());

            SimulateResponse.Builder response = SimulateResponse.newBuilder();
            response.setLeafObservation(result.leafObservation);
            response.setGameEnded(result.gameEnded);
            response.setTerminalReward(result.terminalReward);
            response.setDecisionsPlayed(result.decisionsPlayed);

            responseObserver.onNext(response.build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            responseObserver.onError(
                    io.grpc.Status.INTERNAL
                            .withDescription("Simulate failed: " + e.getMessage())
                            .withCause(e)
                            .asRuntimeException());
        }
    }
}
