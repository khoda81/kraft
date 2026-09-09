//! Reduced ordered multi-way decision diagrams. Every variable is one fixed
//! transition-table entry, with independent uniform destinations in 0..N.

use std::{collections::HashMap, rc::Rc};

use super::GroupedDfaError;

pub(super) type Id = u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) enum Value {
    Index(u32),
    Log(u64),
}

impl Value {
    pub fn log(value: f64) -> Self {
        assert!(value.is_finite());
        Self::Log(if value == 0.0 { 0.0 } else { value }.to_bits())
    }
    pub fn index(self) -> u32 {
        match self {
            Self::Index(value) => value,
            Self::Log(_) => unreachable!("expected an index diagram"),
        }
    }
    pub fn ln(self) -> f64 {
        match self {
            Self::Log(bits) => f64::from_bits(bits),
            Self::Index(_) => unreachable!("expected a log-weight diagram"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) enum Node {
    Leaf(Value),
    Branch { variable: u32, children: Rc<[Id]> },
}

#[derive(Clone, Debug)]
pub(super) struct Diagram {
    pub nodes: Vec<Node>,
    unique: HashMap<Node, Id>,
    pub arity: u16,
    pub limit: usize,
    pub visits: u64,
}

impl Diagram {
    pub fn new(arity: u16, limit: usize) -> Self {
        Self {
            nodes: Vec::new(),
            unique: HashMap::new(),
            arity,
            limit,
            visits: 0,
        }
    }

    fn intern(&mut self, node: Node) -> Result<Id, GroupedDfaError> {
        if let Some(&id) = self.unique.get(&node) {
            return Ok(id);
        }
        if self.nodes.len() >= self.limit {
            return Err(GroupedDfaError::NodeLimit(self.limit));
        }
        let id =
            Id::try_from(self.nodes.len()).map_err(|_| GroupedDfaError::NodeLimit(self.limit))?;
        self.nodes.push(node.clone());
        self.unique.insert(node, id);
        Ok(id)
    }

    pub fn leaf(&mut self, value: Value) -> Result<Id, GroupedDfaError> {
        self.intern(Node::Leaf(value))
    }

    fn branch(&mut self, variable: u32, children: Vec<Id>) -> Result<Id, GroupedDfaError> {
        debug_assert_eq!(children.len(), usize::from(self.arity));
        if children.iter().all(|&child| child == children[0]) {
            return Ok(children[0]);
        }
        debug_assert!(
            children
                .iter()
                .all(|&child| self.top(child).is_none_or(|v| v > variable))
        );
        self.intern(Node::Branch {
            variable,
            children: children.into(),
        })
    }

    pub fn variable(&mut self, variable: u32) -> Result<Id, GroupedDfaError> {
        let children = (0..u32::from(self.arity))
            .map(|i| self.leaf(Value::Index(i)))
            .collect::<Result<_, _>>()?;
        self.branch(variable, children)
    }

    pub fn top(&self, id: Id) -> Option<u32> {
        match self.nodes[id as usize] {
            Node::Leaf(_) => None,
            Node::Branch { variable, .. } => Some(variable),
        }
    }

    pub fn value(&self, id: Id) -> Option<Value> {
        match self.nodes[id as usize] {
            Node::Leaf(value) => Some(value),
            Node::Branch { .. } => None,
        }
    }

    pub fn cofactor(&self, id: Id, variable: u32, destination: usize) -> Id {
        match &self.nodes[id as usize] {
            Node::Branch {
                variable: v,
                children,
            } if *v == variable => children[destination],
            _ => id,
        }
    }

    pub fn map(
        &mut self,
        root: Id,
        mut transform: impl FnMut(Value) -> Result<Value, GroupedDfaError>,
    ) -> Result<Id, GroupedDfaError> {
        fn go(
            manager: &mut Diagram,
            root: Id,
            transform: &mut impl FnMut(Value) -> Result<Value, GroupedDfaError>,
            memo: &mut HashMap<Id, Id>,
        ) -> Result<Id, GroupedDfaError> {
            if let Some(&result) = memo.get(&root) {
                return Ok(result);
            }
            manager.visits += 1;
            let result = match manager.nodes[root as usize].clone() {
                Node::Leaf(value) => manager.leaf(transform(value)?)?,
                Node::Branch { variable, children } => {
                    let children = children
                        .iter()
                        .map(|&child| go(manager, child, transform, memo))
                        .collect::<Result<_, _>>()?;
                    manager.branch(variable, children)?
                }
            };
            memo.insert(root, result);
            Ok(result)
        }
        go(self, root, &mut transform, &mut HashMap::new())
    }

    pub fn add_logs(&mut self, a: Id, b: Id) -> Result<Id, GroupedDfaError> {
        fn go(
            manager: &mut Diagram,
            a: Id,
            b: Id,
            memo: &mut HashMap<(Id, Id), Id>,
        ) -> Result<Id, GroupedDfaError> {
            let key = (a.min(b), a.max(b));
            if let Some(&result) = memo.get(&key) {
                return Ok(result);
            }
            manager.visits += 1;
            let result = if let (Some(a), Some(b)) = (manager.value(a), manager.value(b)) {
                manager.leaf(Value::log(a.ln() + b.ln()))?
            } else {
                let variable = manager
                    .top(a)
                    .into_iter()
                    .chain(manager.top(b))
                    .min()
                    .unwrap();
                let children = (0..usize::from(manager.arity))
                    .map(|d| {
                        go(
                            manager,
                            manager.cofactor(a, variable, d),
                            manager.cofactor(b, variable, d),
                            memo,
                        )
                    })
                    .collect::<Result<_, _>>()?;
                manager.branch(variable, children)?
            };
            memo.insert(key, result);
            Ok(result)
        }
        go(self, a, b, &mut HashMap::new())
    }

    /// Pointwise table lookup. Reduces and memoizes symbolic alternatives; never
    /// enumerates complete transition assignments.
    pub fn select(&mut self, index: Id, choices: &[Id]) -> Result<Id, GroupedDfaError> {
        fn go(
            manager: &mut Diagram,
            index: Id,
            choices: &[Id],
            memo: &mut HashMap<(Id, Vec<Id>), Id>,
        ) -> Result<Id, GroupedDfaError> {
            if choices.iter().all(|&id| id == choices[0]) {
                return Ok(choices[0]);
            }
            if let Some(value) = manager.value(index) {
                return Ok(choices[value.index() as usize]);
            }
            let key = (index, choices.to_vec());
            if let Some(&result) = memo.get(&key) {
                return Ok(result);
            }
            manager.visits += 1;
            let variable = manager
                .top(index)
                .into_iter()
                .chain(choices.iter().filter_map(|&id| manager.top(id)))
                .min()
                .unwrap();
            let children = (0..usize::from(manager.arity))
                .map(|d| {
                    let choices: Vec<_> = choices
                        .iter()
                        .map(|&id| manager.cofactor(id, variable, d))
                        .collect();
                    go(
                        manager,
                        manager.cofactor(index, variable, d),
                        &choices,
                        memo,
                    )
                })
                .collect::<Result<_, _>>()?;
            let result = manager.branch(variable, children)?;
            memo.insert(key, result);
            Ok(result)
        }
        go(self, index, choices, &mut HashMap::new())
    }

    pub fn log_expectation(&self, root: Id, memo: &mut HashMap<Id, f64>) -> f64 {
        if let Some(&value) = memo.get(&root) {
            return value;
        }
        let value = match &self.nodes[root as usize] {
            Node::Leaf(value) => value.ln(),
            Node::Branch { children, .. } => {
                super::log_sum(
                    children
                        .iter()
                        .map(|&child| self.log_expectation(child, memo)),
                ) - f64::from(self.arity).ln()
            }
        };
        memo.insert(root, value);
        value
    }

    pub fn rollback(&mut self, checkpoint: usize) {
        self.nodes.truncate(checkpoint);
        self.unique.retain(|_, id| (*id as usize) < checkpoint);
    }

    /// Rebuild just the live DAG. Shared children remain shared and roots are
    /// remapped together. Collection does not change the represented functions.
    pub fn compact(&self, roots: &[Id]) -> Result<(Self, Vec<Id>), GroupedDfaError> {
        fn copy(
            old: &Diagram,
            new: &mut Diagram,
            id: Id,
            memo: &mut HashMap<Id, Id>,
        ) -> Result<Id, GroupedDfaError> {
            if let Some(&id) = memo.get(&id) {
                return Ok(id);
            }
            let result = match &old.nodes[id as usize] {
                Node::Leaf(value) => new.leaf(*value)?,
                Node::Branch { variable, children } => {
                    let children = children
                        .iter()
                        .map(|&id| copy(old, new, id, memo))
                        .collect::<Result<_, _>>()?;
                    new.branch(*variable, children)?
                }
            };
            memo.insert(id, result);
            Ok(result)
        }
        let mut new = Self::new(self.arity, self.limit);
        new.visits = self.visits;
        let mut memo = HashMap::new();
        let roots = roots
            .iter()
            .map(|&id| copy(self, &mut new, id, &mut memo))
            .collect::<Result<_, _>>()?;
        Ok((new, roots))
    }
}
