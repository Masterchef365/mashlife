use std::str::FromStr;

/// Ruleset for a Life-like cellular automaton
#[derive(Copy, Clone, Debug)]
pub struct Rules {
    /// If a cell is alive and has (index) neighbors, does it survive?
    pub survive: [bool; 9],
    /// If a cell is dead and has (index) neighbors, is it born?
    pub born: [bool; 9],
}

impl Rules {
    /// Query the ruleset with the given number of live neighbors and center state. 
    /// True is alive and False is dead.
    pub fn execute(&self, center: bool, neighbors: usize) -> bool {
        if center {
            self.survive[neighbors]
        } else {
            self.born[neighbors]
        }
    }
}

impl Default for Rules {
    /// Default rule is Conway's Game of Life (B3/S23)
    fn default() -> Self {
        Self::from_str("B3/S23").unwrap()
    }
}

impl FromStr for Rules {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split('/');
        let born = parts
            .next()
            .ok_or_else(|| "No slash in rulestring".to_string())?
            .trim_start_matches('B');
        let survive = parts
            .next()
            .ok_or_else(|| "Empty rule".to_string())?
            .trim_start_matches('S');

        let to_rule_array = |s: &str| -> Result<[bool; 9], String> {
            let mut rules = [false; 9];
            for c in s.chars() {
                if c.is_digit(10) {
                    let dig = c as u8 - b'0';
                    let dig = dig as usize;
                    if dig < rules.len() {
                        rules[dig] = true;
                    } else {
                        return Err(format!("{dig} is not valid in a rule string"));
                    }
                }
            }
            Ok(rules)
        };

        Ok(Self {
            survive: to_rule_array(survive)?,
            born: to_rule_array(born)?,
        })
    }
}
