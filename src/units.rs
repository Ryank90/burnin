//! Parsing and formatting of durations and byte sizes.

use std::time::Duration;

const MIB: f64 = (1u64 << 20) as f64;
const GIB: f64 = (1u64 << 30) as f64;

/// Parses a duration such as `90`, `90s`, `10m` or `2h`. A bare number is seconds.
pub fn parse_duration(text: &str) -> Result<Duration, String> {
    let text = text.trim();
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let value: f64 = number
        .parse()
        .map_err(|_| format!("invalid duration `{text}`"))?;
    let scale = match unit.trim() {
        "" | "s" | "sec" | "secs" => 1.0,
        "m" | "min" | "mins" => 60.0,
        "h" | "hr" | "hrs" => 3600.0,
        other => return Err(format!("unknown duration unit `{other}` (use s, m or h)")),
    };
    let secs = value * scale;
    if !secs.is_finite() || secs <= 0.0 {
        return Err(format!("duration must be greater than zero, got `{text}`"));
    }
    Ok(Duration::from_secs_f64(secs))
}

/// Formats a duration for progress output, e.g. `12.3s`, `4m 05s` or `2h 03m`.
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 60.0 {
        return format!("{secs:.1}s");
    }
    let whole = duration.as_secs();
    if whole < 3600 {
        format!("{}m {:02}s", whole / 60, whole % 60)
    } else {
        format!("{}h {:02}m", whole / 3600, (whole % 3600) / 60)
    }
}

/// Formats a byte count in binary units, e.g. `512 MiB` or `79.6 GiB`.
pub fn format_bytes(bytes: u64) -> String {
    let value = bytes as f64;
    if value >= GIB {
        format!("{:.1} GiB", value / GIB)
    } else {
        format!("{:.0} MiB", value / MIB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_durations() {
        assert_eq!(parse_duration("90"), Ok(Duration::from_secs(90)));
        assert_eq!(parse_duration("90s"), Ok(Duration::from_secs(90)));
        assert_eq!(parse_duration("10m"), Ok(Duration::from_secs(600)));
        assert_eq!(parse_duration("2h"), Ok(Duration::from_secs(7200)));
        assert_eq!(parse_duration("1.5m"), Ok(Duration::from_secs(90)));
        assert_eq!(parse_duration(" 5 min "), Ok(Duration::from_secs(300)));
    }

    #[test]
    fn rejects_bad_durations() {
        assert!(parse_duration("").is_err());
        assert!(parse_duration("0").is_err());
        assert!(parse_duration("10d").is_err());
        assert!(parse_duration("m").is_err());
        assert!(parse_duration("-5s").is_err());
    }

    #[test]
    fn formats_durations() {
        assert_eq!(format_duration(Duration::from_millis(12_340)), "12.3s");
        assert_eq!(format_duration(Duration::from_secs(245)), "4m 05s");
        assert_eq!(format_duration(Duration::from_secs(7380)), "2h 03m");
    }

    #[test]
    fn formats_bytes() {
        assert_eq!(format_bytes(512 << 20), "512 MiB");
        assert_eq!(format_bytes(80 << 30), "80.0 GiB");
        assert_eq!(format_bytes(3 << 29), "1.5 GiB");
    }
}
