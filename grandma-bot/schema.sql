create table grandmas (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  phone text unique not null,
  created_at timestamptz default now()
);

create table family_members (
  id uuid primary key default gen_random_uuid(),
  grandma_id uuid references grandmas(id),
  name text,
  phone text unique not null,
  created_at timestamptz default now()
);

create table memories (
  id uuid primary key default gen_random_uuid(),
  grandma_id uuid references grandmas(id),
  submitted_by_family_id uuid references family_members(id),
  image_url text,
  original_caption text,
  ai_summary text,
  ai_tags jsonb,
  people_mentioned jsonb,
  emotion_hints jsonb,
  era text,
  used_in_sessions jsonb default '[]',
  created_at timestamptz default now()
);

create table sessions (
  id uuid primary key default gen_random_uuid(),
  grandma_id uuid references grandmas(id),
  memory_id uuid references memories(id),
  status text default 'active',
  started_at timestamptz default now(),
  ended_at timestamptz
);

create table turns (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references sessions(id),
  role text,
  content text,
  image_url text,
  created_at timestamptz default now()
);

create table grandma_profile_facts (
  id uuid primary key default gen_random_uuid(),
  grandma_id uuid references grandmas(id),
  fact text,
  source_session_id uuid references sessions(id),
  created_at timestamptz default now()
);

create table session_alerts (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references sessions(id),
  alert_type text not null,
  grandma_message text,
  created_at timestamptz default now()
);

-- seed Margaret + Sarah — REPLACE PHONES BEFORE DEMO
insert into grandmas (id, name, phone) values
  ('11111111-1111-1111-1111-111111111111', 'Margaret', '+15550000001');

insert into family_members (grandma_id, name, phone) values
  ('11111111-1111-1111-1111-111111111111', 'Sarah', '+15550000002');
