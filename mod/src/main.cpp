// GD AI Bridge
//
// Opens a TCP server on 127.0.0.1:22222. A Python agent connects and drives the game in
// lockstep: the game only advances when the agent asks for a step, and every step lasts
// exactly 1/60 s of game time (= 4 physics ticks at 240 Hz). This makes the game
// deterministic and lets it run much faster than real time.
//
// Protocol (one text line per message, "\n"-terminated):
//   agent -> game                game -> agent
//   STEP 0 | STEP 1              S x y vy ground dead won percent mode
//   RESET                        S ...   (state right after the restart)
//   STATE                        S ...   (current state, no time passes)
//   SPEED n                      OK      (n steps per rendered frame, 1 = real time)
//   PRACTICE 1 | PRACTICE 0      OK      (enter / leave practice mode)
//   CHECKPOINT                   OK n    (place a checkpoint here; n = number of checkpoints)
//   CLEARCP                      OK      (remove all checkpoints: RESET goes back to the start)
//   LEVEL id                     OK      (open official level `id`: 1 = Stereo Madness ... 22 = Dash)
//
// In practice mode, RESET respawns at the last checkpoint placed with CHECKPOINT. The checkpoints
// GD places by itself (auto-checkpoints) are removed as soon as they appear.
//
// When no agent is connected, the game behaves exactly as usual.

#include <Geode/Geode.hpp>
#include <Geode/modify/AppDelegate.hpp>
#include <Geode/modify/CCScheduler.hpp>
#include <Geode/modify/PlayLayer.hpp>

// Sockets: Geode is built with precompiled headers that already pull <windows.h>, which itself
// includes the classic <winsock.h>. Including <winsock2.h> after that would redefine everything,
// so we only use functions that <winsock.h> already provides (linked from ws2_32).
#include <winsock.h>

#include <algorithm>
#include <cstdlib>
#include <string>

using namespace geode::prelude;

namespace bridge {

constexpr u_short kPort = 22222;
constexpr float kStepDt = 1.f / 60.f;     // one agent decision = 1/60 s of game time
constexpr DWORD kRecvTimeoutMs = 50;      // keep the window responsive while the agent thinks

SOCKET listenSock = INVALID_SOCKET;
SOCKET client = INVALID_SOCKET;
std::string inbox;
int stepsPerFrame = 20;                   // game steps per rendered frame (speedhack)
bool holding = false;                     // is the jump button currently held by the agent
bool won = false;
unsigned int ourCheckpoints = 0;          // checkpoints placed on the agent's request
bool trimming = false;                    // true while WE remove checkpoints
bool waitingForLayer = false;             // a LEVEL command is loading a new level
PlayLayer* freshLayer = nullptr;          // the PlayLayer created for that new level
int settleFrames = 0;                     // frames to let a freshly opened level start by itself

bool connected() {
    return client != INVALID_SOCKET;
}

void startServer() {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        log::error("WSAStartup failed");
        return;
    }
    listenSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSock == INVALID_SOCKET) {
        log::error("socket() failed");
        return;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(kPort);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // 127.0.0.1 only: not reachable from the network
    if (bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR ||
        listen(listenSock, 1) == SOCKET_ERROR) {
        log::error("Cannot listen on 127.0.0.1:{}", kPort);
        closesocket(listenSock);
        listenSock = INVALID_SOCKET;
        return;
    }
    u_long nonBlocking = 1;
    ioctlsocket(listenSock, FIONBIO, &nonBlocking);   // accept() must never freeze the game
    log::info("AI bridge listening on 127.0.0.1:{}", kPort);
}

void releaseButton(PlayLayer* pl) {
    if (pl && holding) {
        pl->handleButton(false, 1, true);
    }
    holding = false;
}

void dropClient(PlayLayer* pl) {
    releaseButton(pl);
    closesocket(client);
    client = INVALID_SOCKET;
    inbox.clear();
    log::info("AI client disconnected");
}

void tryAccept() {
    if (listenSock == INVALID_SOCKET || connected()) return;
    SOCKET s = accept(listenSock, nullptr, nullptr);
    if (s == INVALID_SOCKET) return;                  // nobody is waiting

    u_long blocking = 0;
    ioctlsocket(s, FIONBIO, &blocking);
    DWORD timeout = kRecvTimeoutMs;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char const*>(&timeout), sizeof(timeout));
    BOOL noDelay = TRUE;                              // send small messages immediately
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char const*>(&noDelay), sizeof(noDelay));

    client = s;
    inbox.clear();
    holding = false;
    won = false;
    ourCheckpoints = 0;
    log::info("AI client connected");
}

enum class ReadResult { Line, Timeout, Closed };

ReadResult readLine(std::string& line) {
    while (true) {
        auto pos = inbox.find('\n');
        if (pos != std::string::npos) {
            line = inbox.substr(0, pos);
            inbox.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return ReadResult::Line;
        }
        char buf[512];
        int n = recv(client, buf, sizeof(buf), 0);
        if (n > 0) {
            inbox.append(buf, n);
            continue;
        }
        if (n == SOCKET_ERROR && WSAGetLastError() == WSAETIMEDOUT) return ReadResult::Timeout;
        return ReadResult::Closed;
    }
}

bool sendLine(std::string const& s) {
    size_t sent = 0;
    while (sent < s.size()) {
        int n = send(client, s.data() + sent, static_cast<int>(s.size() - sent), 0);
        if (n == SOCKET_ERROR) return false;
        sent += n;
    }
    return true;
}

int gamemode(PlayerObject* p) {
    if (p->m_isShip) return 1;
    if (p->m_isBall) return 2;
    if (p->m_isBird) return 3;      // UFO
    if (p->m_isDart) return 4;      // wave
    if (p->m_isRobot) return 5;
    if (p->m_isSpider) return 6;
    if (p->m_isSwing) return 7;
    return 0;                       // cube
}

// Remove every checkpoint the agent did not ask for (GD's automatic ones).
void trimCheckpoints(PlayLayer* pl) {
    auto arr = pl->m_checkpointArray;
    while (arr && arr->count() > ourCheckpoints) {
        auto before = arr->count();
        trimming = true;
        pl->removeCheckpoint(false);                  // false = remove the last one
        trimming = false;
        if (arr->count() >= before) break;            // safety: never loop forever
    }
}

std::string stateLine(PlayLayer* pl) {
    auto p = pl->m_player1;
    bool hasWon = won || pl->m_levelEndAnimationStarted;
    return fmt::format("S {:.4f} {:.4f} {:.4f} {} {} {} {:.3f} {}\n",
        p->m_position.x, p->m_position.y, p->m_yVelocity,
        static_cast<int>(p->m_isOnGround), static_cast<int>(p->m_isDead), static_cast<int>(hasWon),
        pl->getCurrentPercent(), gamemode(p));
}

} // namespace bridge

$on_mod(Loaded) {
    bridge::startServer();
}

class $modify(BridgeScheduler, CCScheduler) {
    void update(float dt) {
        using namespace bridge;
        tryAccept();

        auto pl = PlayLayer::get();
        if (waitingForLayer) {
            // After a LEVEL command: let the game run normally until the new level exists
            if (!freshLayer || pl != freshLayer) {
                CCScheduler::update(dt);
                return;
            }
            waitingForLayer = false;
            // The new PlayLayer exists before the level has really started (its UI is not
            // there yet): resetting it right away crashed GD in PlayLayer::updateTimeLabel.
            // Let it run normally for about 1.5 s first.
            settleFrames = 90;
        }
        if (settleFrames > 0) {
            settleFrames--;
            CCScheduler::update(dt);
            return;
        }
        if (!connected() || !pl || pl->m_isPaused) {
            CCScheduler::update(dt);          // no agent: normal game
            return;
        }

        // Lockstep: answer the agent's commands until we have done `stepsPerFrame` steps,
        // then let GD render one frame.
        int steps = 0;
        while (steps < stepsPerFrame) {
            std::string line;
            auto res = readLine(line);
            if (res == ReadResult::Timeout) return;          // agent busy: the game waits for it
            if (res == ReadResult::Closed) {
                dropClient(pl);
                CCScheduler::update(dt);
                return;
            }

            std::string reply;
            if (line.rfind("STEP", 0) == 0) {
                bool press = line.size() > 5 && line[5] == '1';
                if (press != holding) {
                    pl->handleButton(press, 1, true);         // button 1 = jump, player 1
                    holding = press;
                }
                CCScheduler::update(kStepDt);                 // advance exactly 1/60 s
                pl = PlayLayer::get();
                if (!pl) return;                              // level was left
                trimCheckpoints(pl);
                ++steps;
                reply = stateLine(pl);
            }
            else if (line == "RESET") {
                releaseButton(pl);
                won = false;
                trimCheckpoints(pl);
                pl->resetLevel();
                reply = stateLine(pl);
            }
            else if (line == "PRACTICE 1" || line == "PRACTICE 0") {
                bool on = line.back() == '1';
                releaseButton(pl);
                if (pl->m_isPracticeMode != on) pl->togglePracticeMode(on);
                ourCheckpoints = 0;
                trimCheckpoints(pl);
                reply = "OK\n";
            }
            else if (line == "CHECKPOINT") {
                if (!pl->m_isPracticeMode) {
                    reply = "E not in practice mode\n";
                } else {
                    trimCheckpoints(pl);
                    pl->markCheckpoint();
                    auto n = pl->m_checkpointArray ? pl->m_checkpointArray->count() : 0;
                    if (n > ourCheckpoints) {
                        ourCheckpoints = n;
                        reply = fmt::format("OK {}\n", n);
                    } else {
                        reply = "E checkpoint refused\n";
                    }
                }
            }
            else if (line.rfind("LEVEL", 0) == 0) {
                int id = std::atoi(line.c_str() + 5);
                auto level = GameLevelManager::sharedState()->getMainLevel(id, false);
                if (!level || id < 1) {
                    reply = "E unknown level\n";
                } else {
                    releaseButton(pl);
                    ourCheckpoints = 0;
                    won = false;
                    waitingForLayer = true;
                    freshLayer = nullptr;
                    CCDirector::sharedDirector()->replaceScene(PlayLayer::scene(level, false, false));
                    // The current level is going away: answer, then stop handling commands until
                    // the new level is ready (see the top of this function).
                    if (!sendLine("OK\n")) dropClient(pl);
                    return;
                }
            }
            else if (line == "CLEARCP") {
                ourCheckpoints = 0;
                trimCheckpoints(pl);
                reply = "OK\n";
            }
            else if (line == "STATE") {
                reply = stateLine(pl);
            }
            else if (line.rfind("SPEED", 0) == 0) {
                stepsPerFrame = std::clamp(std::atoi(line.c_str() + 5), 1, 1000);
                reply = "OK\n";
            }
            else {
                reply = "E unknown command\n";
            }

            if (!sendLine(reply)) {
                dropClient(pl);
                return;
            }
        }
    }
};

class $modify(BridgePlayLayer, PlayLayer) {
    // Remember the PlayLayer created after a LEVEL command, to know when the new level is ready
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
        if (bridge::waitingForLayer) bridge::freshLayer = this;
        return true;
    }

    // After a death GD schedules its own restart about 1 s later. The agent restarts the
    // level itself, so this delayed restart would hit in the middle of the next attempt.
    void delayedResetLevel() {
        if (bridge::connected()) return;
        PlayLayer::delayedResetLevel();
    }

    // GD pauses the level when its window loses focus (alt-tab). While the agent is playing,
    // ignore that automatic pause so training keeps running in the background.
    // A manual pause (Escape) still works: GD calls this with unfocused = false.
    void pauseGame(bool unfocused) {
        if (unfocused && bridge::connected()) return;
        PlayLayer::pauseGame(unfocused);
    }

    // In practice mode, GD deletes a checkpoint when the player dies shortly after placing it.
    // The agent dies a lot right after respawning (that is where it explores), so GD would delete
    // our checkpoints and send the next attempts back to the start. Only we remove checkpoints.
    void removeCheckpoint(bool first) {
        if (bridge::connected() && !bridge::trimming) return;
        PlayLayer::removeCheckpoint(first);
    }

    // Do not open the end screen while the agent is playing: just remember the win.
    void levelComplete() {
        if (bridge::connected()) {
            bridge::won = true;
            return;
        }
        PlayLayer::levelComplete();
    }
};

// When the window loses focus or is minimized, GD is told it "goes to the background": it
// pauses the level, pauses the sound and may stop its main loop entirely. While the agent is
// playing, we ignore these notifications so training keeps running behind other windows.
class $modify(BridgeAppDelegate, AppDelegate) {
    void applicationWillResignActive() {
        if (bridge::connected()) return;
        AppDelegate::applicationWillResignActive();
    }

    void applicationDidEnterBackground() {
        if (bridge::connected()) return;
        AppDelegate::applicationDidEnterBackground();
    }
};
