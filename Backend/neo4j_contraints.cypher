CREATE CONSTRAINT candidate_id_unique
IF NOT EXISTS
FOR (c:Candidate)
REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT skill_name_unique
IF NOT EXISTS
FOR (s:Skill)
REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT company_name_unique
IF NOT EXISTS
FOR (cmp:Company)
REQUIRE cmp.name IS UNIQUE;

CREATE CONSTRAINT trait_name_unique
IF NOT EXISTS
FOR (t:Trait)
REQUIRE t.name IS UNIQUE;
