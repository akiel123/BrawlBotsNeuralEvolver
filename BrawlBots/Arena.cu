#include "Arena.h"
#include "Bot.h"
#include <math.h>
#include <curand.h>
#include <ctime>
#include <math.h>


__device__ thrust::device_vector<SSighting> AView(SArena a, float posx, float posy, float r, int id) {
	thrust::device_vector<SSighting> s;
	for (int i = 0; i < a.bots.size(); i++) {
		SNeuralBot temp = temp;
		if (temp.id  == id) continue;
		float angle = getAngle(posx, posy, temp.posx, temp.posy, r);
		if (fmod(fabsf(angle), M_2PI) < VIEW_ANGLE ) {
			float d = distanceTo(posx, posy, temp.posx, temp.posy);
			s.push_back(SSighting(d, angle,  GetVelX(temp), GetVelY(temp),
				ViewType::ENEMY, temp.id, temp.posx, temp.posy));
		}
	}
	for (int i = 0; i < a.ammo.size(); i++) {
		SAmmo temp = temp;
		float angle = getAngle(posx, posy, temp.posx, temp.posy, r);
		if (fmod(fabsf(angle), M_2PI) < VIEW_ANGLE) {
			float d = distanceTo(posx, posy, temp.posx, temp.posy);
			s.push_back(SSighting(d, angle, 0, 0,
				ViewType::AMMO, temp.id, temp.posx, temp.posy));
		}
	}
	for (int i = 0; i < a.bullets.size(); i++) {
		SBullet temp = temp;
		float angle = getAngle(posx, posy, temp.posx, temp.posy, r);
		if (fmod(fabsf(angle), M_2PI) < VIEW_ANGLE) {
			float d = distanceTo(posx, posy, temp.posx, temp.posy);
			s.push_back(SSighting(d, angle, temp.velx, temp.vely,
				ViewType::AMMO, temp.id, temp.posx, temp.posy));
		}
	}
	return s;
}

__device__ float getAngle(float x1, float y1, float x2, float y2, float r1) {
	float vecprod = x1*x2 + y1*y2,
		v1len = hypot(x1, y1),
		v2len = hypot(x2, y2),
		cosAngle = vecprod / (v1len*v2len),
		angle = acos(cosAngle);
	return angle;
}

__device__ float distanceTo(int x1, int y1, int x2, int y2) {
	return hypot(x2 - x1, y2 - y1);
}

__device__ void AUpdate(SArena a) {
	//Spawn ammo?
	if (randomPromille() <= AMMO_SPAWN_PROMILLE / 1000.0) {
		ASpawnAmmo(a);
	}
	//Handle bullets
	thrust::device_vector<SBullet> survivingBullets;
	for (int i = 0; i < a.bullets.size(); i++) {
		SBullet temp = a.bullets[i];
		BUpdate(temp);
		if (temp.posx >= BULLET_RADIUS || temp.posx >= ARENAW + BULLET_RADIUS,
			temp.posy >= BULLET_RADIUS || temp.posy >= ARENAH + BULLET_RADIUS) {
			survivingBullets.push_back(temp);
		}
	}
	a.bullets = survivingBullets;

	//Handle bots
	for (int i = 0; i < a.bots.size(); i++) {
		SNeuralBot temp = a.bots[i];
		BotUpdate(temp, *temp.nnwrk, a);
		
		//X
		if (temp.posx <= MARGIN) {
			temp.posx = MARGIN;
		}
		else if (temp.posx >= ARENAW - BOT_RADIUS - MARGIN) {
			temp.posx = ARENAW - BOT_RADIUS - MARGIN;
		}

		//Y
		if (temp.posy <= MARGIN) {
			temp.posy = MARGIN;
		}
		else if (temp.posy >= ARENAH - BOT_RADIUS - MARGIN) {
			temp.posy = ARENAH - BOT_RADIUS - MARGIN;
		}

		//Speed
		if (temp.vel > MAX_VELOCITY) {
			temp.vel = MAX_VELOCITY;
		}
	}

	//Bullet Collisions
	thrust::device_vector<SBullet> survivingBullets1;
	for (int i = 0; i < a.bullets.size(); i++) {
		int survived = 1;
		for (int j = 0; j < a.bots.size(); j++) {
			SNeuralBot temp = a.bots[j];
			if (BotCollides(temp, a.bullets[i])) {
				temp.health--;
				survived = 0;
			}
		}
		if (survived) {
			survivingBullets1.push_back(a.bullets[i]);
		}
	}
	//Surviving bots
	thrust::device_vector<SNeuralBot> survivingBots;
	for (int i = 0; i < a.bots.size(); i++) {
		SNeuralBot temp = a.bots[i];
		if (temp.health > 0) survivingBots.push_back(a.bots[i]);
	}
	a.bots = survivingBots;

	//Ammo pickups
	thrust::device_vector<SAmmo> survivingAmmo;
	for (int i = 0; i < a.ammo.size(); i++) {
		int survives = 1;
		for (int j = 0; j < a.bots.size(); j++) {
			SNeuralBot temp = a.bots[j];
			if (BotCollides(a.bots[j], a.ammo[i])) {
				temp.ammo++;
				survives = 0;
			}
		}
		if (survives) survivingAmmo.push_back(a.ammo[i]);
	}
	a.ammo = survivingAmmo;

	//End
	a.steps--;
	SNeuralBot temp = a.bots[0];
	if (a.bots.size() < 2 || a.steps <= 0) {
		if (a.bots.size() != 1) a.winner = -2;
		else a.winner = temp.id;
	}
}

__device__ void ASpawnAmmo(SArena a) {
	if (a.ammo.size() >= MAX_AMMO) return;
	SAmmo newA = SAmmo(random(ARENAW - (MARGIN + BOT_RADIUS) * 2) + MARGIN + BOT_RADIUS, 
		random(ARENAH - (MARGIN + BOT_RADIUS) * 2) + MARGIN + BOT_RADIUS);
	a.ammo.push_back(newA);
}

__device__ void AShootFrom(SArena a, float px, float py) {

}

__device__ void BUpdate(SBullet b) {
	b.posx += b.velx;
	b.posy += b.vely;
}

__device__ float randomPromille() {
	return ((clock() * 6234) % 1000) * 1.0 / 1000;
}

__device__ int random(int max) {
	return ((clock() * 6234) % max);
}
__device__ float random(float max) {
	return ((clock() * 6234) % (int)max);
}