"""개발용 실행 도구: 파일을 저장하면 시뮬레이션을 새 코드로 자동 재실행한다.

사용법 (PowerShell, 프로젝트 폴더에서):
    .\\.venv\\Scripts\\python.exe dev_run.py pallet_reach_demo.py --once
    .\\.venv\\Scripts\\python.exe dev_run.py pallet_demo.py 1 0 3 2
    .\\.venv\\Scripts\\python.exe dev_run.py --kill-existing pallet_reach_demo.py   # 따로 떠 있는 시뮬레이션 창을 먼저 닫음
    .\\.venv\\Scripts\\python.exe dev_run.py --manual pallet_reach_demo.py          # 저장해도 재실행하지 않고 R 키로만 재실행

동작:
  - 프로젝트 폴더의 .py 파일과 models/ 아래 .xml 파일을 0.5초마다 확인한다. (dev_run.py 자신은 제외)
  - 저장이 끝나고 0.8초 동안 추가 저장이 없으면 한 번만 처리한다. (여러 번 연달아 저장해도 재실행은 1번)
  - 바뀐 파이썬 파일에 문법 오류가 있으면, 실행 중인 시뮬레이션은 그대로 두고 오류만 출력한다.
  - 문제가 없으면 실행 중인 시뮬레이션을 종료하고 새 코드로 다시 실행한다.
  - 터미널에서 R = 지금 재실행, Q = 종료 (Ctrl+C도 종료)

!! 주의: 재실행은 시뮬레이션을 처음부터 다시 시작한다.
   로봇 자세, 상자 위치, 진행 중이던 단계, 뷰어 카메라 시점이 모두 초기화된다.
"""

import base64
import ctypes
import msvcrt
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOCK_FILE = PROJECT_DIR / ".dev_run.lock"
POLL_INTERVAL = 0.5   # 파일 변경 확인 간격 [s]
QUIET_TIME = 0.8      # 마지막 저장 후 이만큼 조용하면 처리 [s]
STOP_TIMEOUT = 3.0    # 정상 종료를 기다리는 시간 [s], 넘으면 강제 종료
RESTART_GAP = 1.0     # 이전 뷰어 종료 후 새 뷰어 실행까지 쉬는 시간 [s] (메모리·그래픽 자원 정리)
LOW_MEMORY_GB = 1.5   # 새로 확보할 수 있는 메모리가 이보다 적으면 경고 (뷰어 1개에 약 0.6~0.9 GB 필요)
EARLY_EXIT_TIME = 20  # 실행 후 이 시간 안에 오류로 끝나면 시작 실패로 보고 원인 힌트 출력 [s]


def log(message):
    print(f"[dev_run {datetime.now():%H:%M:%S}] {message}", flush=True)


# ----------------------------------------------------------- 파일 감시 --
def watched_files():
    files = [p for p in PROJECT_DIR.glob("*.py") if p.name != "dev_run.py"]
    files += list((PROJECT_DIR / "models").rglob("*.xml"))
    return files


def snapshot():
    """감시 대상 파일별 마지막 수정 시각."""
    result = {}
    for path in watched_files():
        try:
            result[path] = path.stat().st_mtime_ns
        except OSError:
            pass  # 저장 도중 파일이 잠깐 사라지거나 잠길 수 있음 -> 다음 확인 때 다시 봄
    return result


def wait_until_quiet(current):
    """저장이 여러 번 이어지는 동안 기다렸다가, QUIET_TIME 동안 변화가 없으면 최종 상태를 돌려준다."""
    quiet_since = time.time()
    while time.time() - quiet_since < QUIET_TIME:
        time.sleep(0.2)
        latest = snapshot()
        if latest != current:
            current = latest
            quiet_since = time.time()
    return current


def syntax_errors(paths):
    """바뀐 .py 파일을 실행하지 않고 문법만 검사한다."""
    errors = []
    for path in paths:
        if path.suffix != ".py" or not path.exists():
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
        except SyntaxError as e:
            errors.append(f"{path.name} {e.lineno}번째 줄: {e.msg}")
        except (OSError, UnicodeDecodeError) as e:
            errors.append(f"{path.name}: 파일을 읽지 못함 ({e})")
    return errors


# -------------------------------------------------------- 프로세스 관리 --
class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_gb():
    """(남은 RAM, 새로 확보할 수 있는 메모리) [GB]. 내장 그래픽은 그래픽 메모리도 RAM을 나눠 쓴다."""
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 2**30, status.ullAvailPageFile / 2**30


def start_sim(sim_args, count):
    free_ram, free_commit = memory_gb()
    log(f"메모리: 남은 RAM {free_ram:.1f} GB, 새로 확보 가능 {free_commit:.1f} GB")
    if free_commit < LOW_MEMORY_GB:
        log(f"  경고: 메모리가 부족해 뷰어가 시작 중에 실패할 수 있습니다. 다른 뷰어 창이나 브라우저 탭을 닫아 주세요.")
    log(f"시뮬레이션 실행 #{count}: python {' '.join(sim_args)}")
    log("※ 새로 실행하면 시뮬레이션 상태(로봇 자세, 상자 위치, 진행 단계, 카메라 시점)는 처음부터 다시 시작합니다.")
    # 새 프로세스 그룹으로 실행: 터미널의 Ctrl+C가 시뮬레이션에 직접 가지 않고,
    # dev_run이 Ctrl+Break 신호로 시뮬레이션만 골라서 종료할 수 있다.
    return subprocess.Popen([sys.executable, "-u", *sim_args], cwd=PROJECT_DIR,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


def stop_sim(proc):
    """실행 중인 시뮬레이션을 종료한다. 먼저 정상 종료 신호, 안 되면 프로세스 트리 강제 종료."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        proc.wait(STOP_TIMEOUT)
        log(f"이전 시뮬레이션 종료 (PID {proc.pid}, 종료 신호)")
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    # venv의 python.exe는 실제 파이썬을 자식으로 띄우므로 /T로 자식까지 함께 종료한다.
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        pass
    log(f"이전 시뮬레이션 강제 종료 (PID {proc.pid}, taskkill)")


def pid_alive(pid):
    """Windows에서 해당 PID 프로세스가 살아 있는지 확인한다."""
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def acquire_lock():
    """dev_run이 두 개 동시에 실행되지 않도록 잠금 파일을 만든다."""
    if LOCK_FILE.exists():
        try:
            other = int(LOCK_FILE.read_text().strip())
        except ValueError:
            other = None
        if other and pid_alive(other):
            log(f"이미 다른 dev_run이 실행 중입니다 (PID {other}). 그 터미널에서 Q로 종료한 뒤 다시 실행하세요.")
            sys.exit(1)
    LOCK_FILE.write_text(str(os.getpid()))


def release_lock():
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip() == str(os.getpid()):
            LOCK_FILE.unlink()
    except OSError:
        pass


def other_sim_processes():
    """dev_run 밖에서 실행 중인 이 프로젝트 시뮬레이션 (PID, 명령) 목록."""
    script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "ForEach-Object { \"$($_.ProcessId)`t$($_.ParentProcessId)`t$($_.CommandLine)\" }")
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-EncodedCommand", encoded],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    names = {p.name for p in PROJECT_DIR.glob("*.py")} - {"dev_run.py"}
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        pid, parent, cmd = int(parts[0]), int(parts[1]), parts[2]
        if pid in (os.getpid(), os.getppid()) or "dev_run.py" in cmd:
            continue
        if any(name in cmd for name in names):
            rows.append((pid, parent, cmd))
    # venv 런처와 실제 파이썬이 쌍으로 보이므로, 부모가 목록에 없는 것(최상위)만 남긴다.
    pids = {r[0] for r in rows}
    return [r for r in rows if r[1] not in pids]


def read_key():
    """터미널에서 눌린 키 (없으면 None). 뷰어 창에 초점이 있을 때는 뷰어가 키를 받는다."""
    try:
        if msvcrt.kbhit():
            return msvcrt.getwch().lower()
    except OSError:
        pass
    return None


# ------------------------------------------------------------------ 메인 --
def main():
    args = sys.argv[1:]
    kill_existing = "--kill-existing" in args
    manual = "--manual" in args
    sim_args = [a for a in args if a not in ("--kill-existing", "--manual")]
    if not sim_args or not (PROJECT_DIR / sim_args[0]).is_file():
        print(__doc__)
        sys.exit(1)

    acquire_lock()
    try:
        others = other_sim_processes()
        for pid, _, cmd in others:
            if kill_existing:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                log(f"따로 실행 중이던 시뮬레이션을 닫음 (PID {pid})")
            else:
                log(f"주의: dev_run 밖에서 실행 중인 시뮬레이션이 있습니다 (PID {pid}): {cmd}")
        if others and not kill_existing:
            log("  뷰어가 겹쳐 보일 수 있습니다. 직접 닫거나 --kill-existing 옵션으로 다시 실행하세요.")

        if manual:
            log("수동 모드: 저장해도 재실행하지 않습니다 | 터미널에서 R = 문법 검사 후 재실행, Q = 종료")
        else:
            log(f"감시 중: 프로젝트의 .py 파일 {sum(p.suffix == '.py' for p in watched_files())}개, models/ 아래 .xml 파일")
            log("파일을 저장하면 자동 재실행 | 터미널에서 R = 지금 재실행, Q = 종료")
        count = 1
        proc = start_sim(sim_args, count)
        started_at = time.time()
        files = snapshot()
        exit_reported = False

        while True:
            time.sleep(POLL_INTERVAL)
            key = read_key()
            if key == "q":
                break
            reason = "R 키 입력" if key == "r" else None
            if reason and manual:
                errors = syntax_errors([p for p in watched_files() if p.suffix == ".py"])
                if errors:
                    log("문법 오류가 있어 재실행하지 않습니다:")
                    for e in errors:
                        log(f"  {e}")
                    reason = None

            current = files if manual else snapshot()
            if current != files:
                current = wait_until_quiet(current)
                changed = sorted({p for p in files.keys() | current.keys() if files.get(p) != current.get(p)})
                files = current
                if changed:
                    names = ", ".join(p.name for p in changed)
                    errors = syntax_errors(changed)
                    if errors:
                        log(f"저장 감지 ({names}) -> 문법 오류가 있어 재실행하지 않습니다. 실행 중인 시뮬레이션은 그대로 둡니다.")
                        for e in errors:
                            log(f"  {e}")
                    else:
                        reason = f"저장 감지 ({names})"

            if reason:
                log(f"{reason} -> 재실행합니다.")
                stop_sim(proc)
                time.sleep(RESTART_GAP)  # 이전 뷰어의 메모리·그래픽 자원이 풀릴 시간
                count += 1
                proc = start_sim(sim_args, count)
                started_at = time.time()
                exit_reported = False
            elif proc.poll() is not None and not exit_reported:
                log(f"시뮬레이션이 끝났습니다 (종료 코드 {proc.returncode}). 파일을 저장하거나 R을 누르면 다시 실행합니다.")
                if proc.returncode != 0 and time.time() - started_at < EARLY_EXIT_TIME:
                    free_ram, free_commit = memory_gb()
                    log(f"  시작 직후 오류로 끝났습니다. 위 오류 메시지를 확인하세요. "
                        f"(현재 남은 RAM {free_ram:.1f} GB, 확보 가능 {free_commit:.1f} GB)")
                    log("  'Could not allocate memory', 'framebuffer' 오류라면 메모리 부족일 가능성이 큽니다.")
                exit_reported = True
    except KeyboardInterrupt:
        pass
    finally:
        stop_sim(proc if "proc" in locals() else None)
        release_lock()
        log("dev_run 종료")


if __name__ == "__main__":
    main()
