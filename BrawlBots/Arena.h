#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>
#include "NeuralNetwork.h"
#include <thrust\device_vector.h>

#define ARENADIAG 1
#define M_PI 3.14159265359
#define M_2PI 6.28318530718
#define VIEW_ANGLE 1
#define MAX_VELOCITY 1
#define ROTATION_SPEED 1 
#define MAX_AMMO 25
#define AMMO_SPAWN_PROMILLE 2
#define MARGIN 50
#define BOT_RADIUS 16
#define BULLET_RADIUS 5
#define AMMO_RADIUS 10
#define ACCELERATION 1
#define RELOAD 50 
#define STEP_AMOUNT 20000
#define START_HEALTH 6
#define ARENAH 1
#define ARENAW 1
#define START_AMMO 10

enum class ViewType { FRIEND, ENEMY, WALL, BULLET, AMMO };

struct DSighting {
	float distance;
	float angle;
	float velocityx;
	float velocityy;
	ViewType t;
	int id;
	float posx;
	float posy;
	
	DSighting() {};
	DSighting(float d, float r, float vx, float vy, ViewType vt, int ID, float px, float py)
		: distance(d), angle(r), velocityx(vx), velocityy(vy), t(vt), id(ID), posx(px), posy(py) {}
};

struct DSightingBatch {
	DSighting* sights;
	int nSightings;
	DSightingBatch(DSighting * sightings, int numberOfSights) : sights(sightings), nSightings(numberOfSights) {}
};

struct DNeuralBot {
	DNeuralNetwork *nnwrk;
	int reload;
	int NNid;
	float posx;
	float posy;
	float vel;
	float rot;
	int id;
	int ammo;
	int health;
	float* lastOutput;
	__device__ DNeuralBot() {
		nnwrk = &DNeuralNetwork(1, 1);
	}
	__device__ DNeuralBot(DNeuralNetwork* nn) {
		nnwrk = nn;
		id = rand() / RAND_MAX;
		ammo = START_AMMO;
		posx = rand() / RAND_MAX % ARENAW;
		posy = rand() / RAND_MAX % ARENAH;
		rot = fmodf(rand() / RAND_MAX, M_2PI);
	}
};

struct DAmmo {
	float posx;
	float posy;
	int id;
	__device__ DAmmo() {
		id = rand() / RAND_MAX	;
	}
	__device__ DAmmo(float iposx, float iposy) {
		posx = iposx;
		posy = iposy;
		id = rand() / RAND_MAX;
	}
};

struct DBullet {
	float posx;
	float posy;
	float prevPosx;
	float prevPosy;
	float velx;
	float vely;
	int id;
};

struct DArena {
	int winner = -1;
	int steps = 0;
	int botCount;
	int ammoCount;
	int bulletCount;
	DNeuralBot *bots;
	DAmmo *ammo;
	DBullet *bullets;

	__device__ DArena(DNeuralBot b1, DNeuralBot b2) {
		steps = STEP_AMOUNT;
		bots[0] = b1;
		bots[1] = b2;
	}
};




__device__ DSightingBatch AView(DArena a, float posx, float posy, float r, int id);
__device__ float getAngle(float x1, float y1, float x2, float y2, float r1);
__device__ float distanceTo(int x1, int y1, int x2, int y2);
__device__ void AUpdate(DArena a);
__device__ void ASpawnAmmo(DArena a);
__device__ void BUpdate(DBullet b);
__device__ void AShootFrom(DArena a, float px, float py);
__device__ float randomPromille();
__device__ int random(int max);
__device__ float random(float max);