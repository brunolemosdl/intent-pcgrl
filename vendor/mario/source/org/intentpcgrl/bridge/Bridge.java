package org.intentpcgrl.bridge;

import engine.core.MarioAgent;
import engine.core.MarioGame;
import engine.core.MarioResult;
import engine.helper.GameStatus;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class Bridge {
    public static void main(String[] args) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            String line = reader.readLine();
            if (line == null || line.isBlank()) {
                printError("Missing stdin JSON payload");
                System.exit(1);
            }
            JSONObject payload = new JSONObject(line);
            String level = payload.optString("level", "");
            String agentName = payload.optString("agent", "robinBaumgarten");
            int timerSeconds = payload.optInt("timer_seconds", 30);
            int marioState = payload.optInt("mario_state", 0);
            boolean visuals = payload.optBoolean("visuals", false);
            int fps = payload.optInt("fps", 0);
            float scale = (float) payload.optDouble("scale", 2.0);

            MarioAgent agent = loadAgent(agentName);
            MarioGame game = new MarioGame();
            MarioResult result = game.runGame(agent, level, timerSeconds, marioState, visuals, fps, scale);
            printResult(result);
            System.exit(0);
        } catch (Exception ex) {
            printError(ex.getMessage());
            System.exit(1);
        }
    }

    private static MarioAgent loadAgent(String agentName) throws Exception {
        String className = "agents." + agentName + ".Agent";
        Class<?> clazz = Class.forName(className);
        return (MarioAgent) clazz.getDeclaredConstructor().newInstance();
    }

    private static void printResult(MarioResult result) {
        GameStatus status = result.getGameStatus();
        JSONObject out = new JSONObject();
        out.put("ok", true);
        out.put("game_status", status.name());
        out.put("completion_percentage", result.getCompletionPercentage());
        out.put("remaining_time", result.getRemainingTime());
        out.put("mario_mode", result.getMarioMode());
        out.put("kills_total", result.getKillsTotal());
        out.put("kills_by_stomp", result.getKillsByStomp());
        out.put("kills_by_fire", result.getKillsByFire());
        out.put("kills_by_shell", result.getKillsByShell());
        out.put("kills_by_fall", result.getKillsByFall());
        out.put("num_jumps", result.getNumJumps());
        out.put("max_x_jump", result.getMaxXJump());
        out.put("max_jump_air_time", result.getMaxJumpAirTime());
        out.put("current_lives", result.getCurrentLives());
        out.put("current_coins", result.getCurrentCoins());
        out.put("num_collected_mushrooms", result.getNumCollectedMushrooms());
        out.put("num_collected_fireflower", result.getNumCollectedFireflower());
        out.put("num_collected_tile_coins", result.getNumCollectedTileCoins());
        out.put("num_destroyed_bricks", result.getNumDestroyedBricks());
        out.put("num_bump_question_block", result.getNumBumpQuestionBlock());
        out.put("num_bump_brick", result.getNumBumpBrick());
        out.put("num_hurts", result.getMarioNumHurts());
        System.out.println(out);
    }

    private static void printError(String message) {
        JSONObject out = new JSONObject();
        out.put("ok", false);
        out.put("error", message == null ? "unknown error" : message);
        System.out.println(out);
    }
}
