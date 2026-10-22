from loglens.data.normalize import mask
from loglens.data.parse import parse_line


def test_parse_openssh_line():
    rec = parse_line(
        "openssh",
        "Dec 10 06:55:46 LabSZ sshd[24200]: Invalid user webmaster from 173.234.31.186",
    )
    assert rec is not None
    assert rec.host == "LabSZ"
    assert rec.component == "sshd"
    assert rec.message == "Invalid user webmaster from 173.234.31.186"


def test_parse_linux_component_with_parens():
    rec = parse_line(
        "linux",
        "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0",
    )
    assert rec is not None
    assert rec.component == "sshd(pam_unix)"
    assert rec.message.startswith("authentication failure")


def test_parse_apache_keeps_level_in_message():
    rec = parse_line(
        "apache",
        "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6",
    )
    assert rec is not None
    assert rec.level == "error"
    assert rec.message == "[error] mod_jk child workerEnv in error state 6"


def test_parse_bgl_with_alert_flag():
    rec = parse_line(
        "bgl",
        "KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-17.32.07.581048 "
        "R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt",
    )
    assert rec is not None
    assert rec.alert is True
    assert rec.level == "FATAL"
    assert rec.message == "KERNEL FATAL data TLB error interrupt"


def test_parse_bgl_non_alert_and_empty_message():
    ok = parse_line(
        "bgl",
        "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 "
        "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
    )
    assert ok is not None and ok.alert is False
    empty = parse_line(
        "bgl", "- 1117838570 2005.06.03 R02 2005-06-03-15.42.50.363779 R02 RAS KERNEL INFO"
    )
    assert empty is None


def test_unparseable_line_returns_none():
    assert parse_line("apache", "garbage") is None
    assert parse_line("linux", "   ") is None


def test_ids_are_stable_and_source_scoped():
    line = "[Sun Dec 04 04:47:44 2005] [error] boom"
    assert parse_line("apache", line).id == parse_line("apache", line).id


def test_mask_collapses_variables():
    a = mask("Invalid user admin from 1.2.3.4 port 5555")
    b = mask("Invalid user admin from 10.0.0.9 port 42")
    assert a == b == "invalid user admin from <ip> port <num>"


def test_mask_hdfs_blocks_and_paths():
    m = mask("Receiving block blk_-123 src: /10.250.9.207:59759 dest: /var/lib/hdfs/data")
    assert "<blk>" in m and "<ip>" in m and "<path>" in m


def test_mask_register_dumps():
    a = mask("KERNEL FATAL fpr2=0x0f5b9b03 3e458839 d9d7bdbb 3ddb7cdf")
    b = mask("KERNEL FATAL fpr2=0x11111111 22222222 abcdef12 3ddb7cd0")
    assert a == b


def test_mask_keeps_plain_words():
    assert "accepted" in mask("Accepted password for root")
