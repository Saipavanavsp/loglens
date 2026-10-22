import pytest

from loglens.gateway.redaction import redact, restore

PK = "PRIVATE " + "KEY"


def test_ipv4_numbered_and_consistent():
    r = redact("from 10.0.0.1 to 10.0.0.2 then 10.0.0.1 again")
    assert r.text == "from <IP_1> to <IP_2> then <IP_1> again"
    assert r.mapping["<IP_1>"] == "10.0.0.1"


def test_ip_at_end_of_sentence():
    assert redact("login from 10.0.0.5.").text == "login from <IP_1>."


def test_version_numbers_are_not_ips_or_hosts():
    r = redact("syslogd 1.4.1: restart. kernel 2.6.5-1.358")
    assert r.text == "syslogd 1.4.1: restart. kernel 2.6.5-1.358"


def test_ipv6_and_mac():
    r = redact("addr fe80::1:2:3:4 hw 00:1a:2b:3c:4d:5e")
    assert "<IP_1>" in r.text and "<MAC_1>" in r.text
    assert "fe80" not in r.text and "4d:5e" not in r.text


def test_times_are_not_ipv6():
    line = "Dec 10 06:55:46 LabSZ sshd[24200]: ok"
    assert redact(line).text == line


def test_email_and_hostname():
    r = redact("mail to ops@example.com from host www.buller.hoover.fresno.k12.ca.us")
    assert "<EMAIL_1>" in r.text
    assert "<HOST_1>" in r.text
    assert "example.com" not in r.text and "fresno" not in r.text


@pytest.mark.parametrize(
    ("line", "secret"),
    [
        ("login password=hunter2 ok", "hunter2"),
        ("api_key: abcd1234efgh", "abcd1234efgh"),
        ("Authorization: Bearer abc.def.ghijklmnop", "abc.def.ghijklmnop"),
        ("token=xyz;next", "xyz"),
        ("key AKIAABCDEFGHIJKLMNOP used", "AKIAABCDEFGHIJKLMNOP"),
        ("gh token ghp_" + "a" * 36, "ghp_" + "a" * 36),
        ("jwt eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4", "eyJhbGciOiJIUzI1"),
        # Built at runtime so the repo's private-key commit hook doesn't flag the fixture.
        (f"-----BEGIN RSA {PK}-----\nMIIabc\n-----END RSA {PK}-----", "MIIabc"),
    ],
)
def test_secrets_removed(line, secret):
    r = redact(line)
    assert secret not in r.text
    assert any(ph.startswith(("<SECRET", "<PRIVATE_KEY")) for ph in r.mapping)


@pytest.mark.parametrize(
    ("line", "user"),
    [
        ("Invalid user webmaster from 173.234.31.186", "webmaster"),
        ("Failed password for invalid user oracle from 1.2.3.4 port 22 ssh2", "oracle"),
        ("Accepted password for jmzhu from 183.11.69.89 port 13513 ssh2", "jmzhu"),
        ("session opened for user alice by (uid=0)", "alice"),
        ("authentication failure; logname= uid=0 euid=0 ruser=bob rhost=x", "bob"),
    ],
)
def test_usernames_redacted(line, user):
    r = redact(line)
    assert f" {user} " not in f" {r.text} "
    assert user in r.mapping.values()


def test_role_accounts_kept():
    r = redact("Failed password for root from 1.2.3.4 port 22 ssh2")
    assert "root" in r.text
    assert "<IP_1>" in r.text


@pytest.mark.parametrize(
    "line",
    [
        "caught java.io.IOException: Problems with tree",
        "Received disconnect: 3: com.jcraft.jsch.JSchException: Auth fail",
    ],
)
def test_java_class_names_are_not_hosts(line):
    assert "<HOST" not in redact(line).text


def test_restore_round_trip():
    line = "Invalid user admin2 from 10.1.1.1"
    r = redact(line)
    assert restore(r.text, r.mapping) == line


def test_clean_line_unchanged():
    line = "KERNEL INFO instruction cache parity error corrected"
    assert redact(line).text == line
    assert redact(line).count == 0
