#include "Arena.h"
#include "Bot.h"
#include <curand.h>
#include <ctime>
#include <math.h>


__device__ DSightingBatch AView(DArena a, float posx, float posy, float r, int id) {
	DSighting* s;
	int count;
	for (int i = 0; i < a.botCount; i++, count++) {
		DNeuralBot temp = temp;
		if (temp.id  == id) continue;
		float angle = getAngle(posx, posy, temp.posx, temp.posy, r);
		if (fmod((float)fabsf(angle), (float)M_2PI) < VIEW_ANGLE ) {
			float d = distanceTo(posx, posy, temp.posx, temp.posy);
			s[count] = DSighting(d, angle,  GetVelX(temp), GetVelY(temp),
				ViewType::ENEMY, temp.id, temp.posx, temp.posy);
		}
	}
	for (int i = 0; i < a.ammoCount; i++, count++) {
		DAmmo temp = temp;
		float angle = getAngle(posx, posy, temp.posx, temp.posy, r);
		if (fmod((float)fabsf(angle), (float)M_2PI) < VIEW_ANGLE) {
			float d = distanceTo(posx, posy, temp.posx, temp.posy);
			s[count] = DSighting(d, angle, 0, 0,
				ViewType::AMMO, temp.id, temp.posx, temp.posy);
		}
	}
	for (int i = 0; i < a.bulletCount; i++, count++) {
		DBullet temp = temp;
		float angle = getAngle(posx, posy, temp.posx, temp.posy, r);
		if (fmod((float)fabsf(angle), (float)M_2PI) < VIEW_ANGLE) {
			float d = distanceTo(posx, posy, temp.posx, temp.posy);
			s[count] = DSighting(d, angle, temp.velx, temp.vely,
				ViewType::AMMO, temp.id, temp.posx, temp.posy);
		}
	}
	DSightingBatch b = DSightingBatch(s, count);
	return b;
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
	return hypotf((float)(x2 - x1), (float)(y2 - y1));
}

__device__ void AUpdate(DArena a) {
	//Spawn ammo?
	if (randomPromille() <= AMMO_SPAWN_PROMILLE / 1000.0) {
		ASpawnAmmo(a);
	}
	//Handle bullets
	DBullet* survivingBullets;
	int count = 0;
	for (int i = 0; i < a.bulletCount; i++, count++) {
		DBullet temp = a.bullets[i];
		BUpdate(temp);
		/*if (temp.posx >= BULLET_RADIUS || temp.posx >= ARENAW + BULLET_RADIUS, //Should this be checking for surviving bullets?
			temp.posy >= BULLET_RADIUS || temp.posy >= ARENAH + BULLET_RADIUS) {
			survivingBullets[i] = temp;
		}*/
	}
	a.bullets = survivingBullets;
	a.bulletCount = count;
	//Handle bots
	for (int i = 0; i < a.botCount; i++) {
		DNeuralBot temp = a.bots[i];
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
	DBullet* survivingBullets1;
	count = 0;
	for (int i = 0; i < a.bulletCount; i++) {
		int survived = 1;
		for (int j = 0; j < a.botCount; j++) {
			DNeuralBot temp = a.bots[j];
			if (BotCollides(temp, a.bullets[i])) {
				temp.health--;
				survived = 0;
			}
		}
		if (survived) {
			survivingBullets1[count] = a.bullets[i];
			count++;
		}
	}
	a.bullets = survivingBullets1;
	a.bulletCount = count;

	//Surviving bots
	DNeuralBot* survivingBots;
	count = 0;
	for (int i = 0; i < a.botCount; i++) {
		DNeuralBot temp = a.bots[i];
		if (temp.health > 0) {
			survivingBots[count] = a.bots[i];
			count++;
		}
	}
	a.bots = survivingBots;
	a.botCount = count;

	//Ammo pickups
	DAmmo* survivingAmmo;
	count = 0;
	for (int i = 0; i < a.ammoCount; i++) {
		int survives = 1;
		for (int j = 0; j < a.botCount; j++) {
			DNeuralBot temp = a.bots[j];
			if (BotCollides(a.bots[j], a.ammo[i])) {
				temp.ammo++;
				survives = 0;
			}
		}
		if (survives) {
			survivingAmmo[count] = a.ammo[i];
			count++;
		}
	}
	a.ammo = survivingAmmo;
	a.ammoCount = count;

	//End
	a.steps--;
	DNeuralBot temp = a.bots[0];
	if (a.botCount < 2 || a.steps <= 0) {
		if (a.botCount != 1) a.winner = -2;
		else a.winner = temp.id;
	}
}

__device__ void ASpawnAmmo(DArena a) {
	if (a.ammoCount >= MAX_AMMO) return;
	DAmmo newA = DAmmo(random(ARENAW - (MARGIN + BOT_RADIUS) * 2) + MARGIN + BOT_RADIUS, 
		random(ARENAH - (MARGIN + BOT_RADIUS) * 2) + MARGIN + BOT_RADIUS);
	a.ammo[a.ammoCount] = newA;
	a.ammoCount++;
}

__device__ void AShootFrom(DArena a, float px, float py) {

}

__device__ void BUpdate(DBullet b) {
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