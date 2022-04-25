DROP SCHEMA IF EXISTS nascar CASCADE;
create schema nascar;

drop table if exists nascar.personal_info CASCADE;
drop table if exists nascar.social_media CASCADE;
drop table if exists nascar.seasonal_achievement CASCADE;
drop table if exists nascar.detailed_performance CASCADE;
drop table if exists nascar.earnings CASCADE;
drop table if exists nascar.debut_info CASCADE;
drop table if exists nascar.race_info CASCADE;
drop table if exists nascar.status_info CASCADE;


create table nascar.personal_info (
    id                      serial primary key,
    driver_id               varchar(256),
    first_name              varchar(256),
    last_name               varchar(256),
    full_name               varchar(256),
    gender                  varchar(256),
    height                  float,
    weight                  float,
    birthday                timestamp,
    birth_place             varchar(256),
    country                 varchar(256)
    );

create table nascar.social_media (
    social_media_id           serial primary key,
    driver_id                 int not null references nascar.personal_info("id"),
    twitter                   varchar(256)
    );


create table nascar.race_info (
    id                        serial primary key,
    series                    varchar(256),
    season                    int
    );
    
    
create table nascar.seasonal_achievement(
    seasonal_achievement_id     serial primary key,
    race_id                     int not null references nascar.race_info("id"),
    driver_id                   int not null references nascar.personal_info("id"),
    rank                        int,
    starts                      int,
    points                      int,
    wins                        int,
    stage_wins                  int,
    poles                       int,
    top_5                       int,
    top_10                      int,
    top_15                      int,
    top_20                      int
);

create table nascar.detailed_performance (
    detailed_performance_id   serial primary key,
    race_id                 int not null references nascar.race_info("id"),
    driver_id                 int not null references nascar.personal_info("id"),
    chase_bonus               int,
    chase_wins                int,
    chase_stage_wins          int,
    laps_led                  int,
    laps_completed            int,
    avg_start_position        float,
    avg_finish_position       float,
    avg_laps_completed        float,
    laps_led_pct              float,
    dnf                       int,
    in_chase                  boolean,
    behind                    float
    );

create table nascar.earnings (
    earnings_id                serial primary key,
    driver_id                 int not null references nascar.personal_info("id"),
    race_id                   int not null references nascar.race_info("id"),
    money                     float
    );

create table nascar.debut_info (
    debut_info_id              serial primary key,
    driver_id                 int not null references nascar.personal_info("id"),
    series                    varchar(256),
    rookie_year               int
    );

    
create table nascar.status_info (
    status_info_id            serial primary key,
    driver_id                 int not null references nascar.personal_info("id"),
    status                    varchar(256)
    ); 



